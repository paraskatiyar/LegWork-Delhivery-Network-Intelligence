"""What counts as "late", and why the obvious answer is the wrong one.

The brief defines a chronically delayed corridor as one where actual time
exceeds OSRM by more than 20%. Applied to this network that rule flags ~94% of
legs and ~94% of corridors. A metric that fires on nineteen of every twenty
shipments cannot rank anything, and every reviewed submission that led with it
ended up telling an operations leader that essentially the entire network is
broken -- which is both unhelpful and untrue.

The reason is that the 1.2x rule measures two different things at once:

  1. **Forecast bias.** OSRM models free-flow driving. It has no notion of
     facility dwell, loading, or multi-stop carting. Its median leg estimate is
     ~1.9x optimistic *by construction*. This gap is systematic and therefore
     predictable -- it is fixed with a better ETA model, at zero capex.

  2. **Operational excess.** Variation beyond what that corridor normally
     achieves. This is what congestion, a saturated hub, or a missed dispatch
     wave actually looks like -- and it is the only part a facility upgrade can
     recover.

So we calibrate the promise to the corridor's own demonstrated capability,
learned from training legs only, and call a leg late when it misses that
promise by more than the 15% tolerance the brief already uses for ETA accuracy.
Both definitions are computed and reported side by side; the memo leads with
the calibrated one and shows the raw one for continuity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

TOLERANCE = 0.15  # matches the "within 15% of actual" bar in the brief


@dataclass
class SLACalibration:
    """Corridor-level promise multipliers, fitted on training legs only."""

    corridor_factor: pd.Series          # index: corridor_id -> multiplier on OSRM
    global_factor: float
    shrinkage_k: int
    tolerance: float = TOLERANCE

    @classmethod
    def fit(cls, train_legs: pd.DataFrame, cfg: Config) -> "SLACalibration":
        global_factor = float(train_legs["delay_ratio_w"].median())
        k = cfg.sla.min_legs_for_corridor_promise
        q = cfg.sla.calibration_quantile

        stats = train_legs.groupby("corridor_id", observed=True)["delay_ratio_w"].agg(
            ["size", lambda s: s.quantile(q)]
        )
        stats.columns = ["n", "corridor_q"]
        # Empirical-Bayes shrinkage: a corridor seen twice should not define its
        # own promise; it borrows from the network until it has earned the right.
        shrunk = (stats["n"] * stats["corridor_q"] + k * global_factor) / (stats["n"] + k)
        return cls(corridor_factor=shrunk, global_factor=global_factor, shrinkage_k=k)

    def apply(self, legs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        out = legs.copy()
        factor = out["corridor_id"].map(self.corridor_factor).fillna(self.global_factor)
        out["promise_factor"] = factor
        out["promised_minutes"] = out["osrm_minutes"] * factor

        # Calibrated: late against a promise the network has shown it can keep.
        out["late_threshold_minutes"] = out["promised_minutes"] * (1 + self.tolerance)
        out["is_late"] = (out["actual_minutes"] > out["late_threshold_minutes"]).astype("int8")
        out["excess_minutes"] = (
            out["actual_minutes"] - out["late_threshold_minutes"]
        ).clip(lower=0)

        # Raw: the brief's literal definition, kept for comparability.
        out["is_late_raw"] = (
            out["delay_ratio"] > cfg.sla.ratio_threshold
        ).astype("int8")
        out["excess_minutes_raw"] = (
            out["actual_minutes"] - out["osrm_minutes"] * cfg.sla.ratio_threshold
        ).clip(lower=0)
        return out


def summarise(legs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row contrasting the two definitions -- this table anchors the memo."""
    total = len(legs)
    return pd.DataFrame([
        {
            "definition": f"raw: actual > {cfg.sla.ratio_threshold:g}x OSRM",
            "late_legs": int(legs["is_late_raw"].sum()),
            "late_rate_pct": float(legs["is_late_raw"].mean() * 100),
            "total_excess_minutes": float(legs["excess_minutes_raw"].sum()),
            "usable_for_ranking": "no - fires on almost every leg",
        },
        {
            "definition": (
                f"calibrated: actual > corridor promise x {1 + TOLERANCE:g} "
                "(promise = OSRM x shrunk corridor median, train-fitted)"
            ),
            "late_legs": int(legs["is_late"].sum()),
            "late_rate_pct": float(legs["is_late"].mean() * 100),
            "total_excess_minutes": float(legs["excess_minutes"].sum()),
            "usable_for_ranking": "yes - concentrates on genuine operational excess",
        },
        {
            "definition": "legs evaluated",
            "late_legs": total,
            "late_rate_pct": 100.0,
            "total_excess_minutes": float(legs["delay_minutes"].clip(lower=0).sum()),
            "usable_for_ranking": "-",
        },
    ])


def decompose_delay(legs: pd.DataFrame) -> dict[str, float]:
    """Partition the delay-vs-OSRM gap into its three parts, exactly.

    This is the number that decides the investment sequence: if most of the gap
    is forecast bias, ship the model before you pour concrete. So it has to be a
    real partition, not three separately-computed quantities that happen to be
    near 100% -- the naive version sums past 100 because the tolerance band
    between the promise and the late threshold belongs to neither bucket, and
    because a leg that beats its promise still contributes a full systematic
    share it never actually incurred.

    Per leg, over ``gap = max(actual - osrm, 0)``:

        systematic  = min(max(promise - osrm, 0), gap)      capped at the gap
        tolerance   = min(max(threshold - promise, 0), gap - systematic)
        operational = gap - systematic - tolerance          equals excess_minutes

    The three are non-negative and sum to the gap by construction.
    """
    gap = legs["delay_minutes"].clip(lower=0)
    systematic = (legs["promised_minutes"] - legs["osrm_minutes"]).clip(lower=0)
    systematic = pd.concat([systematic, gap], axis=1).min(axis=1)
    remaining = gap - systematic
    tolerance = (legs["late_threshold_minutes"] - legs["promised_minutes"]).clip(lower=0)
    tolerance = pd.concat([tolerance, remaining], axis=1).min(axis=1)
    operational = (remaining - tolerance).clip(lower=0)

    total_gap = float(gap.sum())
    def share(series: pd.Series) -> float:
        return 100 * float(series.sum()) / total_gap if total_gap else 0.0

    return {
        "total_gap_vs_osrm_minutes": total_gap,
        "systematic_forecast_bias_minutes": float(systematic.sum()),
        "systematic_share_pct": share(systematic),
        "tolerance_band_minutes": float(tolerance.sum()),
        "tolerance_share_pct": share(tolerance),
        "operational_excess_minutes": float(operational.sum()),
        "operational_share_pct": share(operational),
    }
