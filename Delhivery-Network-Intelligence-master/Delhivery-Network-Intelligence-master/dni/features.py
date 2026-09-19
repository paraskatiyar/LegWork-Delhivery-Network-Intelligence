"""Feature assembly with an enforced leakage firewall.

Every feature block is produced by an object that was fitted on training legs
only. The assembled matrix is then checked against `FORBIDDEN_FEATURES` and the
run *aborts* if anything outcome-derived slipped in.

This check is not decoration. In one reviewed submission the graph model was
handed every column except the target, where the target was
`actual_time - osrm_time` and both terms were still present as features -- the
model was solving arithmetic, and the resulting "84% MAE improvement" was
carried into a strategy memo recommending a production rollout. A three-line
assertion would have caught it before it reached the executive summary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .embeddings import BaseEmbedder
from .graph import NetworkModel
from .ingest import FORBIDDEN_FEATURES

#: Known at dispatch: the routing engine's own estimate plus calendar context.
#: This is the honest information set for quoting an ETA.
TRIP_NUMERIC = [
    "osrm_minutes", "osrm_km", "start_hour", "is_weekend",
    "log_osrm_minutes", "log_osrm_km", "implied_speed_kmph",
    "hour_sin", "hour_cos",
]
TRIP_CATEGORICAL = ["route_type", "time_of_day", "day_of_week"]

EDGE_FEATURES = [
    "edge_median_delay_ratio", "edge_mean_delay_ratio", "edge_p75_delay_ratio",
    "edge_leg_count", "edge_median_actual_minutes", "edge_median_osrm_minutes",
    "edge_median_osrm_km", "edge_late_rate", "edge_ftl_share",
    "edge_stratum_delay_ratio", "edge_is_cold_start",
]

HUB_FEATURES = [
    f"{side}_{metric}"
    for side in ("src", "dst")
    for metric in (
        "betweenness", "pagerank", "clustering_coeff", "in_degree", "out_degree",
        "weighted_in_degree", "weighted_out_degree", "sla_contribution_pct",
        "chokepoint_score", "is_cold_start",
    )
]


def add_trip_features(legs: pd.DataFrame) -> pd.DataFrame:
    out = legs.copy()
    out["log_osrm_minutes"] = np.log1p(out["osrm_minutes"])
    out["log_osrm_km"] = np.log1p(out["osrm_km"])
    # OSRM's own implied speed separates a congested urban lane from a highway
    # run of the same duration, without referencing anything post-hoc.
    out["implied_speed_kmph"] = out["osrm_km"] / (out["osrm_minutes"] / 60).clip(lower=1e-6)
    out["hour_sin"] = np.sin(2 * np.pi * out["start_hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["start_hour"] / 24)
    return out


@dataclass
class FeatureSpace:
    """Assembles the four feature blocks the ablation ladder steps through."""

    cfg: Config
    network: NetworkModel
    embedder: BaseEmbedder | None = None
    _embed_columns: list[str] = field(default_factory=list)

    def build(self, legs: pd.DataFrame) -> pd.DataFrame:
        base = add_trip_features(legs)
        blocks = [base, self.network.edge_features(base), self.network.hub_features(base)]
        if self.embedder is not None:
            embedded = self.embedder.frame(base)
            self._embed_columns = list(embedded.columns)
            blocks.append(embedded)
        return pd.concat(blocks, axis=1)

    def columns(self, level: str) -> tuple[list[str], list[str]]:
        """Feature columns for one rung of the ablation ladder.

        The rungs are strictly nested and share an identical trip-feature base,
        so a difference between two rungs is attributable to the block that was
        added -- and to nothing else. Comparing a graph model that receives
        corridor history against a baseline denied it measures the feature, not
        the graph.
        """
        numeric = list(TRIP_NUMERIC)
        if level in ("corridor", "structural", "embedded"):
            numeric += EDGE_FEATURES
        if level in ("structural", "embedded"):
            numeric += HUB_FEATURES
        if level == "embedded":
            numeric += self._embed_columns
        return numeric, list(TRIP_CATEGORICAL)


def assert_no_leakage(matrix: pd.DataFrame, columns: list[str]) -> None:
    """Abort the run if any outcome-derived column reached the model matrix."""
    offenders = sorted(set(columns) & FORBIDDEN_FEATURES)
    if offenders:
        raise AssertionError(
            "Leakage firewall tripped. These columns encode the outcome and are "
            f"not available when an ETA must be quoted: {offenders}. "
            "Remove them from the feature list rather than relaxing this check."
        )
    missing = sorted(set(columns) - set(matrix.columns))
    if missing:
        raise AssertionError(f"Feature columns missing from matrix: {missing}")
