"""The logistics network as a directed, weighted, stratified graph.

Design decisions worth defending:

* **Nodes** are facilities; **edges** are corridors. Edge weight is the median
  actual/OSRM ratio on that corridor -- median rather than mean because a
  single 30x outlier (network outage, mis-scanned shipment) would otherwise
  redefine the corridor.

* **Stratification.** The brief asks for weights stratified by route type and
  time of day. A single scalar per corridor cannot express "this lane is fine
  by day and terrible at night", which is exactly the pattern an operations
  team can act on without capex. We therefore store the collapsed weight as
  `weight` (so any NetworkX algorithm works out of the box) *and* a
  per-stratum dictionary on the edge, which the feature layer reads back.

* **Fit/transform.** `NetworkModel.fit` may only ever see training legs. Test
  legs go through `.transform`, which looks features up and marks anything
  unseen as cold-start rather than inventing a value. This is enforced by the
  class shape rather than by a comment, because "remember to filter to train"
  is exactly the discipline that fails under deadline -- and it failed in four
  of the five reviewed submissions, each of which reported a graph advantage
  that was partly its own test set leaking back in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from .config import Config


@dataclass
class GraphArtifacts:
    graph: nx.DiGraph
    edge_stats: pd.DataFrame
    strata_stats: pd.DataFrame
    hub_metrics: pd.DataFrame


def _edge_frame(legs: pd.DataFrame) -> pd.DataFrame:
    return (
        legs.groupby(["source_center", "destination_center"], observed=True)
        .agg(
            leg_count=("trip_uuid", "size"),
            median_delay_ratio=("delay_ratio_w", "median"),
            mean_delay_ratio=("delay_ratio_w", "mean"),
            p75_delay_ratio=("delay_ratio_w", lambda s: s.quantile(0.75)),
            median_actual_minutes=("actual_minutes", "median"),
            median_osrm_minutes=("osrm_minutes", "median"),
            median_osrm_km=("osrm_km", "median"),
            total_excess_minutes=("excess_minutes", "sum"),
            late_rate=("is_late", "mean"),
            ftl_share=("route_type", lambda s: float((s == "FTL").mean())),
        )
        .reset_index()
    )


def _strata_frame(legs: pd.DataFrame) -> pd.DataFrame:
    strata = (
        legs.groupby(
            ["source_center", "destination_center", "route_type", "time_of_day"],
            observed=True, dropna=False,
        )
        .agg(
            leg_count=("trip_uuid", "size"),
            median_delay_ratio=("delay_ratio_w", "median"),
            late_rate=("is_late", "mean"),
            total_excess_minutes=("excess_minutes", "sum"),
        )
        .reset_index()
    )
    strata["corridor_id"] = (
        strata["source_center"] + "->" + strata["destination_center"]
    )
    return strata


class NetworkModel:
    """Learns network structure from training legs; scores any legs."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.graph: nx.DiGraph | None = None
        self.edge_stats: pd.DataFrame | None = None
        self.strata_stats: pd.DataFrame | None = None
        self.hub_metrics: pd.DataFrame | None = None
        self._edge_lookup: dict[tuple[str, str], dict] = {}
        self._strata_lookup: dict[tuple[str, str, str, str], float] = {}
        self._fitted_on: str | None = None

    # -- fit ----------------------------------------------------------------
    def fit(self, train_legs: pd.DataFrame) -> "NetworkModel":
        if train_legs["split"].nunique() > 1 or train_legs["split"].iloc[0] != "train":
            raise ValueError(
                "NetworkModel.fit received non-training legs. The graph must be "
                "learned from the training split only, or every downstream "
                "'graph advantage' is contaminated by the test set."
            )
        cfg = self.cfg
        edges = _edge_frame(train_legs)
        edges = edges[edges["leg_count"] >= cfg.graph.min_legs_per_edge].copy()
        strata = _strata_frame(train_legs)

        graph = nx.DiGraph()
        names = pd.concat([
            train_legs[["source_center", "source_name"]].rename(
                columns={"source_center": "center", "source_name": "name"}),
            train_legs[["destination_center", "destination_name"]].rename(
                columns={"destination_center": "center", "destination_name": "name"}),
        ], ignore_index=True).dropna()
        name_map = names.groupby("center", observed=True)["name"].agg(
            lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
        )

        strata_by_edge: dict[tuple[str, str], dict[str, float]] = {}
        for row in strata.itertuples(index=False):
            if row.leg_count < cfg.graph.min_legs_per_edge:
                continue
            key = (row.source_center, row.destination_center)
            strata_by_edge.setdefault(key, {})[
                f"{row.route_type}|{row.time_of_day}"
            ] = float(row.median_delay_ratio)

        for row in edges.itertuples(index=False):
            key = (row.source_center, row.destination_center)
            attrs = {
                "weight": float(row.median_delay_ratio),
                "median_delay_ratio": float(row.median_delay_ratio),
                "mean_delay_ratio": float(row.mean_delay_ratio),
                "p75_delay_ratio": float(row.p75_delay_ratio),
                "leg_count": int(row.leg_count),
                "median_actual_minutes": float(row.median_actual_minutes),
                "median_osrm_minutes": float(row.median_osrm_minutes),
                "median_osrm_km": float(row.median_osrm_km),
                "total_excess_minutes": float(row.total_excess_minutes),
                "late_rate": float(row.late_rate),
                "ftl_share": float(row.ftl_share),
                "strata": json.dumps(strata_by_edge.get(key, {})),
            }
            graph.add_edge(*key, **attrs)
            self._edge_lookup[key] = attrs

        for node in graph.nodes():
            graph.nodes[node]["facility_name"] = str(name_map.get(node, node))

        self.graph = graph
        self.edge_stats = edges
        self.strata_stats = strata
        self._strata_lookup = {
            (s, d, rt, tod): v
            for (s, d), inner in strata_by_edge.items()
            for k, v in inner.items()
            for rt, tod in [tuple(k.split("|", 1))]
        }
        self.hub_metrics = self._compute_hub_metrics(train_legs)
        self._fitted_on = "train"
        return self

    # -- structural metrics --------------------------------------------------
    def _compute_hub_metrics(self, train_legs: pd.DataFrame) -> pd.DataFrame:
        g = self.graph
        assert g is not None

        # Betweenness with OSRM minutes as the traversal *cost*. Using the
        # delay ratio as cost (as one reviewed submission did) inverts the
        # intent: shortest paths then avoid slow corridors, so the hubs sitting
        # on the worst lanes score LOWER, which is the opposite of a bottleneck.
        betweenness = nx.betweenness_centrality(
            g, weight="median_osrm_minutes", normalized=True
        )
        undirected = g.to_undirected()
        clustering = nx.clustering(undirected, weight="weight")
        pagerank = nx.pagerank(g, weight="leg_count")

        inbound = (
            train_legs.groupby("destination_center", observed=True)
            .agg(in_legs=("trip_uuid", "size"),
                 in_excess_minutes=("excess_minutes", "sum"),
                 in_late_legs=("is_late", "sum"))
            .rename_axis("facility")
        )
        outbound = (
            train_legs.groupby("source_center", observed=True)
            .agg(out_legs=("trip_uuid", "size"),
                 out_excess_minutes=("excess_minutes", "sum"),
                 out_late_legs=("is_late", "sum"))
            .rename_axis("facility")
        )

        rows = []
        for node, attrs in g.nodes(data=True):
            rows.append({
                "facility": node,
                "facility_name": attrs.get("facility_name", node),
                "betweenness": betweenness.get(node, 0.0),
                "pagerank": pagerank.get(node, 0.0),
                "clustering_coeff": clustering.get(node, 0.0),
                "in_degree": g.in_degree(node),
                "out_degree": g.out_degree(node),
                "weighted_in_degree": g.in_degree(node, weight="leg_count"),
                "weighted_out_degree": g.out_degree(node, weight="leg_count"),
            })
        hubs = pd.DataFrame(rows).set_index("facility")
        hubs = hubs.join(inbound, how="left").join(outbound, how="left").fillna(0)

        hubs["legs_touched"] = hubs["in_legs"] + hubs["out_legs"]
        hubs["late_legs_touched"] = hubs["in_late_legs"] + hubs["out_late_legs"]
        hubs["excess_minutes_touched"] = (
            hubs["in_excess_minutes"] + hubs["out_excess_minutes"]
        )

        # SLA attribution. A leg touches two facilities, so summing this column
        # over all hubs double-counts by construction; we divide by the total
        # *attributions* rather than the total excess so the column reads as a
        # true share and sums to 100%. (Two reviewed submissions reported hub
        # shares that silently summed past 100% for exactly this reason.)
        attribution_total = hubs["excess_minutes_touched"].sum()
        hubs["sla_contribution_pct"] = (
            100 * hubs["excess_minutes_touched"] / attribution_total
            if attribution_total else 0.0
        )

        for col in ["betweenness", "weighted_in_degree", "weighted_out_degree",
                    "excess_minutes_touched"]:
            peak = hubs[col].max()
            hubs[f"{col}_score"] = hubs[col] / peak if peak else 0.0

        # Composite chokepoint score: structural criticality x operational harm.
        # Weights are a judgement call and are stated wherever the score is used.
        hubs["chokepoint_score"] = (
            0.35 * hubs["betweenness_score"]
            + 0.20 * hubs["weighted_in_degree_score"]
            + 0.20 * hubs["weighted_out_degree_score"]
            + 0.25 * hubs["excess_minutes_touched_score"]
        )
        return (
            hubs.reset_index()
            .sort_values(["chokepoint_score", "excess_minutes_touched"], ascending=False)
            .reset_index(drop=True)
        )

    # -- transform -----------------------------------------------------------
    def edge_features(self, legs: pd.DataFrame) -> pd.DataFrame:
        """Attach corridor-level history. Unseen corridors are flagged, not faked."""
        if self._fitted_on is None:
            raise RuntimeError("NetworkModel must be fitted before transform.")
        cols = [
            "median_delay_ratio", "mean_delay_ratio", "p75_delay_ratio", "leg_count",
            "median_actual_minutes", "median_osrm_minutes", "median_osrm_km",
            "late_rate", "ftl_share",
        ]
        blank = {f"edge_{c}": np.nan for c in cols}
        records = []
        for src, dst, rt, tod in zip(
            legs["source_center"], legs["destination_center"],
            legs["route_type"], legs["time_of_day"],
        ):
            attrs = self._edge_lookup.get((src, dst))
            if attrs is None:
                rec = dict(blank)
                rec["edge_is_cold_start"] = 1
            else:
                rec = {f"edge_{c}": attrs[c] for c in cols}
                rec["edge_is_cold_start"] = 0
            rec["edge_stratum_delay_ratio"] = self._strata_lookup.get(
                (src, dst, str(rt), str(tod)), np.nan
            )
            records.append(rec)
        return pd.DataFrame(records, index=legs.index)

    def hub_features(self, legs: pd.DataFrame) -> pd.DataFrame:
        assert self.hub_metrics is not None
        cols = [
            "betweenness", "pagerank", "clustering_coeff", "in_degree", "out_degree",
            "weighted_in_degree", "weighted_out_degree", "sla_contribution_pct",
            "chokepoint_score",
        ]
        table = self.hub_metrics.set_index("facility")[cols]
        out = {}
        for side, key in (("src", "source_center"), ("dst", "destination_center")):
            joined = table.reindex(legs[key].to_numpy())
            for c in cols:
                out[f"{side}_{c}"] = joined[c].to_numpy()
            out[f"{side}_is_cold_start"] = np.isnan(joined["betweenness"].to_numpy()).astype(int)
        return pd.DataFrame(out, index=legs.index)

    def cold_start_report(self, legs: pd.DataFrame) -> dict[str, float]:
        """How much of the holdout is structurally unseen?

        Worth measuring explicitly: an inductive model (GraphSAGE) can serve
        these facilities, a transductive one (node2vec) cannot, and that
        difference is a deployment constraint, not a metric footnote.
        """
        assert self.graph is not None
        known = set(self.graph.nodes())
        src_unseen = ~legs["source_center"].isin(known)
        dst_unseen = ~legs["destination_center"].isin(known)
        edge_unseen = [
            (s, d) not in self._edge_lookup
            for s, d in zip(legs["source_center"], legs["destination_center"])
        ]
        return {
            "legs": len(legs),
            "unseen_source_facilities": int(legs.loc[src_unseen, "source_center"].nunique()),
            "unseen_destination_facilities": int(legs.loc[dst_unseen, "destination_center"].nunique()),
            "legs_with_unseen_endpoint_pct": float(((src_unseen | dst_unseen).mean()) * 100),
            "legs_on_unseen_corridor_pct": float(np.mean(edge_unseen) * 100),
        }


def chronic_corridor_audit(
    legs: pd.DataFrame, network: NetworkModel, cfg: Config, top_n: int | None = None
) -> pd.DataFrame:
    """Rank corridors by SLA-breach contribution, not by raw delay ratio.

    Ranking by ratio surfaces lanes with 3 shipments and a 30x outlier. Ranking
    by contributed excess minutes surfaces the lanes that actually cost the
    network its service level, which is what an operations leader can fund.
    """
    # Cap per-leg excess before aggregating. Without this, one 26x lane running
    # ten shipments outranks a trunk corridor carrying four thousand -- which is
    # how a priority list ends up recommending capex on a scanning error.
    work = legs.copy()
    cap = work["excess_minutes"].quantile(0.99)
    work["excess_capped"] = work["excess_minutes"].clip(upper=cap)

    audit = (
        work.groupby(["source_center", "destination_center"], observed=True)
        .agg(
            legs=("trip_uuid", "size"),
            median_delay_ratio=("delay_ratio", "median"),
            late_rate=("is_late", "mean"),
            late_rate_raw=("is_late_raw", "mean"),
            excess_minutes=("excess_capped", "sum"),
            excess_minutes_uncapped=("excess_minutes", "sum"),
            median_actual_minutes=("actual_minutes", "median"),
            median_osrm_minutes=("osrm_minutes", "median"),
            median_osrm_km=("osrm_km", "median"),
            ftl_share=("route_type", lambda s: float((s == "FTL").mean())),
        )
        .reset_index()
    )
    total_excess = audit["excess_minutes"].sum()
    audit["sla_contribution_pct"] = (
        100 * audit["excess_minutes"] / total_excess if total_excess else 0.0
    )
    audit["is_chronic_raw"] = (audit["median_delay_ratio"] > cfg.sla.ratio_threshold).astype(int)
    audit["is_chronic_calibrated"] = (
        (audit["late_rate"] > 0.5) & (audit["legs"] >= cfg.sla.min_legs_for_corridor_promise)
    ).astype(int)
    # A corridor only reaches the priority list if it carries enough traffic for
    # its statistics to mean anything AND its delay is not an extreme artifact.
    audit["is_credible_priority"] = (
        (audit["legs"] >= cfg.sla.min_legs_for_priority)
        & (audit["median_delay_ratio"] <= cfg.sla.artifact_ratio_ceiling)
    ).astype(int)

    names = pd.concat([
        legs[["source_center", "source_name"]].rename(
            columns={"source_center": "c", "source_name": "n"}),
        legs[["destination_center", "destination_name"]].rename(
            columns={"destination_center": "c", "destination_name": "n"}),
    ]).drop_duplicates("c").set_index("c")["n"]
    audit["source_facility"] = audit["source_center"].map(names)
    audit["destination_facility"] = audit["destination_center"].map(names)
    audit["corridor"] = audit["source_facility"] + " -> " + audit["destination_facility"]

    audit = audit.sort_values("excess_minutes", ascending=False).reset_index(drop=True)
    return audit.head(top_n) if top_n else audit
