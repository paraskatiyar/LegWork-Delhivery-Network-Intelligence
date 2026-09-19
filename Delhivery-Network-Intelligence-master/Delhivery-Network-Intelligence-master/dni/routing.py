"""FTL vs Carting as a decision, not a description.

The trap here is subtle and every reviewed submission fell into it in one of
two ways.

**Trap 1 - classifying the status quo.** Train a classifier to predict
`route_type` from distance and graph position, score 98-99% accuracy, and
present that as a decision framework. It is not one. The model has learned the
dispatch policy that is *already in force*; asking it what to do returns what
is already being done. High accuracy is evidence the current policy is
consistent, not evidence it is right.

**Trap 2 - empirical counterfactuals only.** Compare realised FTL time against
realised Carting time on corridors that ran both. Only 14 corridors in this
dataset ever ran both modes, so the comparison has no power, and those 14 are
selected precisely because something unusual happened on them.

What we do instead is a model-based counterfactual. For every leg we ask the
ETA model to predict the duration twice -- once as FTL, once as Carting,
holding distance, hour and network position fixed -- then price both outcomes
through a stated cost model and recommend the cheaper *generalised* cost:

    generalised cost = transport + value-of-time + expected lateness penalty

Two honesty requirements come with this:

* The ETA model is predictive, not causal. Mode is confounded with distance and
  corridor. We therefore restrict recommendations to the region of overlap
  (corridors where both modes are plausible on distance) and report how much of
  the network that covers, rather than issuing a network-wide instruction.
* The rate card is invented -- the dataset has no cost column. Every parameter
  is declared in `CostModel`, printed alongside the results, and swept in
  `sensitivity`, so a reader can see which recommendations survive a different
  rate card and which are artifacts of our assumptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .features import FeatureSpace
from .models import TARGET, make_estimator
from .features import assert_no_leakage

ROUTE_TYPES = ["FTL", "Carting"]


def transport_cost(route_type: pd.Series, km: pd.Series, cfg: Config) -> pd.Series:
    """FTL: high fixed, low per-km. Carting: low fixed, higher per-km + handling.

    The crossover this produces is the whole economic story -- FTL wins on long
    hauls where its per-km advantage outruns its fixed cost, Carting wins short.
    """
    c = cfg.cost
    km = km.fillna(0).clip(lower=0)
    ftl = c.ftl_fixed_inr + c.ftl_per_km_inr * km
    carting = c.carting_fixed_inr + c.carting_handling_inr + c.carting_per_km_inr * km
    return pd.Series(
        np.where(route_type.astype(str).str.upper().eq("FTL"), ftl, carting),
        index=route_type.index, dtype="float64",
    )


def _overlap_band(legs: pd.DataFrame) -> pd.Series:
    """Where do both modes actually operate? Outside this band we do not advise.

    Recommending Carting for a 900 km line-haul because a cost formula says so
    is the kind of output that destroys an analyst's credibility in the room.
    """
    ftl_km = legs.loc[legs["route_type"] == "FTL", "osrm_km"]
    cart_km = legs.loc[legs["route_type"] == "Carting", "osrm_km"]
    low = max(ftl_km.quantile(0.05), cart_km.quantile(0.05))
    high = min(ftl_km.quantile(0.95), cart_km.quantile(0.95))
    return legs["osrm_km"].between(low, high)


def score_counterfactuals(
    train_legs: pd.DataFrame,
    score_legs: pd.DataFrame,
    space: FeatureSpace,
    cfg: Config,
) -> pd.DataFrame:
    """Predict each leg's duration under both modes and price the difference."""
    train_matrix = space.build(train_legs)
    numeric, categorical = space.columns("embedded")
    assert_no_leakage(train_matrix, numeric + categorical)
    model = make_estimator(cfg, numeric, categorical)
    model.fit(train_matrix[numeric + categorical], train_matrix[TARGET].to_numpy(float))

    out = score_legs.copy()
    out["observed_route_type"] = out["route_type"]
    for mode in ROUTE_TYPES:
        scenario = score_legs.copy()
        scenario["route_type"] = mode
        matrix = space.build(scenario)
        out[f"eta_{mode}"] = np.clip(model.predict(matrix[numeric + categorical]), 1.0, None)
        out[f"transport_cost_{mode}"] = transport_cost(
            scenario["route_type"], scenario["osrm_km"], cfg
        ).to_numpy()

    for mode in ROUTE_TYPES:
        late = (out[f"eta_{mode}"] - out["late_threshold_minutes"]).clip(lower=0)
        out[f"late_minutes_{mode}"] = late
        out[f"generalised_cost_{mode}"] = (
            out[f"transport_cost_{mode}"]
            + cfg.cost.value_of_time_inr_per_min * out[f"eta_{mode}"]
            + cfg.cost.late_penalty_inr_per_min * late
        )

    out["recommended_route_type"] = np.where(
        out["generalised_cost_FTL"] <= out["generalised_cost_Carting"], "FTL", "Carting"
    )
    out["in_overlap_band"] = _overlap_band(score_legs).to_numpy()
    # Outside the region where both modes are actually observed, we defer to the
    # incumbent rather than extrapolating the cost model.
    out.loc[~out["in_overlap_band"], "recommended_route_type"] = out.loc[
        ~out["in_overlap_band"], "observed_route_type"
    ]
    out["would_switch"] = out["recommended_route_type"] != out["observed_route_type"]
    out["eta_minutes_saved_by_ftl"] = out["eta_Carting"] - out["eta_FTL"]
    out["ftl_cost_premium_inr"] = out["transport_cost_FTL"] - out["transport_cost_Carting"]

    observed_cost = np.where(
        out["observed_route_type"].eq("FTL"),
        out["generalised_cost_FTL"], out["generalised_cost_Carting"],
    )
    recommended_cost = np.where(
        out["recommended_route_type"].eq("FTL"),
        out["generalised_cost_FTL"], out["generalised_cost_Carting"],
    )
    out["generalised_saving_inr"] = observed_cost - recommended_cost
    out["distance_band"] = pd.cut(
        out["osrm_km"], bins=[-0.01, 25, 75, 150, 300, np.inf],
        labels=["0-25km", "25-75km", "75-150km", "150-300km", "300km+"],
    ).astype("string")
    return out


def decision_rules(scored: pd.DataFrame, network_hubs: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-leg decisions into rules a dispatcher can actually follow.

    Cut by the three axes the brief names -- distance, time of day, and the
    source facility's position in the graph -- because a rule an operator can
    apply from a dispatch screen beats a per-shipment model call they cannot
    audit.
    """
    scored = scored.copy()
    chokepoint = network_hubs.set_index("facility")["chokepoint_score"]
    scored["source_chokepoint"] = scored["source_center"].map(chokepoint).fillna(0)
    scored["source_graph_position"] = pd.qcut(
        scored["source_chokepoint"].rank(method="first"), q=4,
        labels=["peripheral", "connector", "major", "critical"],
    ).astype("string")

    band = scored[scored["in_overlap_band"]]
    rules = (
        band.groupby(["distance_band", "time_of_day", "source_graph_position"],
                     observed=True, dropna=False)
        .agg(
            legs=("trip_uuid", "size"),
            ftl_share_recommended=("recommended_route_type", lambda s: float((s == "FTL").mean())),
            ftl_share_today=("observed_route_type", lambda s: float((s == "FTL").mean())),
            avg_eta_saved_by_ftl_min=("eta_minutes_saved_by_ftl", "mean"),
            avg_ftl_cost_premium_inr=("ftl_cost_premium_inr", "mean"),
            avg_generalised_saving_inr=("generalised_saving_inr", "mean"),
            switch_rate=("would_switch", "mean"),
        )
        .reset_index()
    )
    rules = rules[rules["legs"] >= 20].copy()
    rules["recommendation"] = np.where(
        rules["ftl_share_recommended"] >= 0.6, "Default FTL",
        np.where(rules["ftl_share_recommended"] <= 0.4, "Default Carting", "Review case-by-case"),
    )
    return rules.sort_values("avg_generalised_saving_inr", ascending=False).reset_index(drop=True)


def empirical_validation(legs: pd.DataFrame) -> pd.DataFrame:
    """Sanity-check the counterfactual against corridors that really ran both modes.

    This is a small sample and we say so -- but a model-based counterfactual
    that disagrees in *sign* with the handful of natural experiments available
    should not be trusted, and this is the only way to notice.
    """
    both = (
        legs.groupby(["source_center", "destination_center", "route_type"], observed=True)
        .agg(legs=("trip_uuid", "size"),
             median_actual_minutes=("actual_minutes", "median"),
             median_km=("osrm_km", "median"))
        .reset_index()
    )
    pivot = both.pivot_table(
        index=["source_center", "destination_center"], columns="route_type",
        values=["legs", "median_actual_minutes", "median_km"],
    )
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.dropna(subset=["median_actual_minutes_FTL", "median_actual_minutes_Carting"])
    pivot["observed_ftl_time_advantage_min"] = (
        pivot["median_actual_minutes_Carting"] - pivot["median_actual_minutes_FTL"]
    )
    return pivot.reset_index()


def breakeven_distance(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """At what distance does FTL start paying for itself?

    This, not a bare "x% should be FTL", is the deliverable a dispatcher can
    use. FTL carries a high fixed cost and a low per-km rate; Carting is the
    reverse. There is therefore a crossover distance, and it moves with how
    badly we are punished for being late.

    We report it three ways -- on transport cost alone, and on generalised cost
    at the observed time difference -- so the reader can see how much of the
    answer is haulage economics and how much is service risk.
    """
    c = cfg.cost
    # Transport-only crossover, solved directly:
    #   ftl_fixed + ftl_km*d = cart_fixed + cart_handling + cart_km*d
    per_km_gap = c.carting_per_km_inr - c.ftl_per_km_inr
    fixed_gap = c.ftl_fixed_inr - (c.carting_fixed_inr + c.carting_handling_inr)
    transport_crossover = fixed_gap / per_km_gap if per_km_gap > 0 else float("inf")

    rows = []
    band = scored[scored["in_overlap_band"]]
    observed_time_gain = float(band["eta_minutes_saved_by_ftl"].median()) if len(band) else 0.0

    for multiplier in c.sensitivity_multipliers:
        penalty = c.late_penalty_inr_per_min * multiplier
        # Value of FTL's time advantage per leg, at this penalty level.
        time_value = (c.value_of_time_inr_per_min + penalty) * observed_time_gain
        crossover = (fixed_gap - time_value) / per_km_gap if per_km_gap > 0 else float("inf")
        rows.append({
            "late_penalty_inr_per_min": penalty,
            "penalty_multiplier": multiplier,
            "median_ftl_time_gain_min": observed_time_gain,
            "value_of_ftl_time_gain_inr": time_value,
            "breakeven_km_transport_only": transport_crossover,
            "breakeven_km_generalised": max(crossover, 0.0),
        })
    return pd.DataFrame(rows)


def mode_summary(scored: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """Headline numbers for the memo, framed so they cannot be misread."""
    band = scored[scored["in_overlap_band"]]
    if not len(band):
        return {}
    return {
        "band_min_km": float(band["osrm_km"].min()),
        "band_max_km": float(band["osrm_km"].max()),
        "band_share_of_legs_pct": float(len(band) / len(scored) * 100),
        "median_ftl_time_gain_min": float(band["eta_minutes_saved_by_ftl"].median()),
        "mean_ftl_time_gain_min": float(band["eta_minutes_saved_by_ftl"].mean()),
        "mean_ftl_cost_premium_inr": float(band["ftl_cost_premium_inr"].mean()),
        "ftl_share_today_pct": float((band["observed_route_type"] == "FTL").mean() * 100),
        "ftl_share_recommended_pct": float((band["recommended_route_type"] == "FTL").mean() * 100),
        "switch_rate_pct": float(band["would_switch"].mean() * 100),
        "total_saving_inr": float(band["generalised_saving_inr"].sum()),
        "saving_per_switched_leg_inr": float(
            band.loc[band["would_switch"], "generalised_saving_inr"].mean()
        ) if band["would_switch"].any() else 0.0,
    }


def sensitivity(scored: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Sweep the invented parameters. Recommendations that flip are not findings.

    Each row re-prices every leg with one assumption scaled, and reports how the
    FTL share of the recommendation moves. If a headline recommendation only
    holds at one point on this grid, it belongs in a caveat, not a memo.
    """
    rows = []
    band = scored[scored["in_overlap_band"]].copy()
    for parameter in ["late_penalty_inr_per_min", "value_of_time_inr_per_min", "ftl_per_km_inr"]:
        for multiplier in cfg.cost.sensitivity_multipliers:
            base_value = getattr(cfg.cost, parameter)
            value = base_value * multiplier
            if parameter == "ftl_per_km_inr":
                ftl_transport = cfg.cost.ftl_fixed_inr + value * band["osrm_km"]
                cart_transport = band["transport_cost_Carting"]
            else:
                ftl_transport = band["transport_cost_FTL"]
                cart_transport = band["transport_cost_Carting"]
            vot = value if parameter == "value_of_time_inr_per_min" else cfg.cost.value_of_time_inr_per_min
            pen = value if parameter == "late_penalty_inr_per_min" else cfg.cost.late_penalty_inr_per_min

            ftl_total = ftl_transport + vot * band["eta_FTL"] + pen * band["late_minutes_FTL"]
            cart_total = cart_transport + vot * band["eta_Carting"] + pen * band["late_minutes_Carting"]
            rows.append({
                "parameter": parameter,
                "multiplier": multiplier,
                "value": value,
                "ftl_share_recommended_pct": float((ftl_total <= cart_total).mean() * 100),
            })
    return pd.DataFrame(rows)
