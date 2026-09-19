"""Revenue-at-risk, and what a hub upgrade actually recovers.

The brief asks for the revenue-at-risk recovered if the top three hubs are
upgraded. That number is only worth writing down if the chain from data to
rupees is visible, so this module keeps three rules:

1. **Every assumption is a named parameter** in `RevenueModel`, printed with
   the result. No figure is sourced to an outside benchmark, because we do not
   have one. One reviewed submission attributed its recovery rates and an NPS
   correlation to "Kearney 2024 benchmarks" that do not exist -- an invented
   citation is worse than an admitted assumption.

2. **Every headline gets a range**, produced by sweeping those parameters, and
   the range is reported in the memo. A point estimate implies a precision the
   inputs cannot support.

3. **We only claim the recoverable part.** ~86% of the gap against OSRM is
   systematic forecast bias: real, expensive, and *not* something a bigger dock
   fixes. Sizing a capex case against the whole gap -- as the reviewed
   submissions did, reaching up to Rs 340 Cr -- overstates the return by
   roughly seven times. A hub upgrade is credited only with operational excess
   at the hubs it touches.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def revenue_at_risk(legs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Value the late legs, low/central/high on the penalty assumption."""
    r = cfg.revenue
    late_legs = int(legs["is_late"].sum())
    rows = []
    for label, share in (
        ("low", r.penalty_share_low),
        ("central", r.penalty_share_of_revenue),
        ("high", r.penalty_share_high),
    ):
        exposure = late_legs * r.revenue_per_leg_inr * share
        rows.append({
            "scenario": label,
            "penalty_share_of_revenue": share,
            "late_legs": late_legs,
            "revenue_per_leg_inr": r.revenue_per_leg_inr,
            "revenue_at_risk_inr": exposure,
            "revenue_at_risk_cr": exposure / 1e7,
            # The dataset is a sample of the network, not the whole of it, so
            # the absolute rupee figure is small by construction. Normalising
            # per 100k legs lets the reader scale it by their own true volume
            # instead of us inventing one.
            "revenue_at_risk_per_100k_legs_inr": exposure / len(legs) * 100_000,
        })
    return pd.DataFrame(rows)


def hub_upgrade_scenario(
    legs: pd.DataFrame, hub_metrics: pd.DataFrame, cfg: Config, top_n: int = 3,
    detail_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate halving operational excess at the top-N chokepoint hubs.

    Mechanically: for every leg touching a target hub, reduce its excess minutes
    by the recovery rate, recompute whether it would still miss its promise, and
    count the legs that flip from late to on-time. That is the number an
    operations leader is being asked to buy.
    """
    r = cfg.revenue
    targets = hub_metrics.head(top_n)
    target_ids = set(targets["facility"])

    touched = legs[
        legs["source_center"].isin(target_ids) | legs["destination_center"].isin(target_ids)
    ]
    baseline_late = int(legs["is_late"].sum())

    # Mechanism matters here. Scaling every leg's excess down by a fixed
    # proportion is arithmetically tidy and operationally meaningless: under
    # that model no leg ever crosses back inside its promise unless the
    # recovery rate is 100%, so the scenario always returns zero.
    #
    # What a hub upgrade actually does is remove a roughly constant slug of
    # dwell -- an extra dock, a second dispatch wave, a faster sort take the
    # same handful of minutes off every shipment through the building. So we
    # size the intervention as a fixed minutes-saved per leg, anchored to the
    # hub's own median excess, and a leg is recovered when that slug covers its
    # shortfall. Legs far beyond the promise stay late, which is correct: one
    # dock does not rescue a shipment running four hours over.
    late_touched = touched[touched["is_late"] == 1]
    anchor = float(late_touched["excess_minutes"].median()) if len(late_touched) else 0.0

    rows = []
    for label, rate in (
        ("conservative", r.recovery_low),
        ("central", r.excess_delay_recovery_rate),
        ("optimistic", r.recovery_high),
    ):
        minutes_removed = anchor * rate
        eligible = touched["excess_minutes"] > 0
        recovered = int((eligible & (touched["excess_minutes"] <= minutes_removed)).sum())
        minutes_recovered = float(
            touched.loc[eligible, "excess_minutes"].clip(upper=minutes_removed).sum()
        )
        for share_label, share in (("low", r.penalty_share_low),
                                   ("central", r.penalty_share_of_revenue),
                                   ("high", r.penalty_share_high)):
            value = recovered * r.revenue_per_leg_inr * share
            rows.append({
                "recovery_scenario": label,
                "excess_recovery_rate": rate,
                "dwell_minutes_removed_per_leg": minutes_removed,
                "penalty_scenario": share_label,
                "penalty_share": share,
                "legs_touching_target_hubs": len(touched),
                "late_legs_touching_target_hubs": len(late_touched),
                "late_legs_before": baseline_late,
                "late_legs_recovered": recovered,
                "late_reduction_pct": 100 * recovered / baseline_late if baseline_late else 0.0,
                "excess_minutes_recovered": minutes_recovered,
                "revenue_recovered_inr": value,
                "revenue_recovered_cr": value / 1e7,
                "revenue_recovered_per_100k_legs_inr": (
                    value / len(legs) * 100_000 if len(legs) else 0.0
                ),
            })

    # Detail covers a wider set than the capital case so the memo can show the
    # next hubs in line without implying they are being funded.
    detail = []
    for _, hub in hub_metrics.head(detail_n).iterrows():
        hub_legs = legs[
            (legs["source_center"] == hub["facility"])
            | (legs["destination_center"] == hub["facility"])
        ]
        detail.append({
            "rank": len(detail) + 1,
            "facility": hub["facility"],
            "facility_name": hub["facility_name"],
            "betweenness": hub["betweenness"],
            "chokepoint_score": hub["chokepoint_score"],
            "sla_contribution_pct": hub["sla_contribution_pct"],
            "legs_touched": len(hub_legs),
            "late_legs": int(hub_legs["is_late"].sum()),
            "late_rate_pct": float(hub_legs["is_late"].mean() * 100) if len(hub_legs) else 0.0,
            "excess_minutes": float(hub_legs["excess_minutes"].sum()),
            "median_delay_ratio": float(hub_legs["delay_ratio"].median()) if len(hub_legs) else 0.0,
            "in_capital_case": bool(hub["facility"] in target_ids),
        })
    return pd.DataFrame(rows), pd.DataFrame(detail)


def eta_model_value(ladder_metrics: pd.DataFrame, legs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Value the software lever: better promises, no concrete.

    If the promise quoted to the customer is the model's prediction rather than
    OSRM's, the share of shipments that miss their quoted window falls to the
    model's own miss rate. This is the cheapest lever available and it addresses
    the 86% of the gap that capex cannot touch.
    """
    osrm_row = ladder_metrics[ladder_metrics["rung"] == "osrm"].iloc[0]
    best_row = ladder_metrics[ladder_metrics["rung"] == "embedded"].iloc[0]
    r = cfg.revenue

    osrm_miss_pct = 100 - float(osrm_row["within_15pct"])
    model_miss_pct = 100 - float(best_row["within_15pct"])
    total_legs = len(legs)
    legs_fixed = total_legs * (osrm_miss_pct - model_miss_pct) / 100

    rows = []
    for label, share in (("low", r.penalty_share_low),
                         ("central", r.penalty_share_of_revenue),
                         ("high", r.penalty_share_high)):
        value = legs_fixed * r.revenue_per_leg_inr * share
        rows.append({
            "penalty_scenario": label,
            "penalty_share": share,
            "osrm_quote_miss_pct": osrm_miss_pct,
            "model_quote_miss_pct": model_miss_pct,
            "promise_miss_reduction_pp": osrm_miss_pct - model_miss_pct,
            "legs_brought_inside_window": legs_fixed,
            "value_inr": value,
            "value_cr": value / 1e7,
            "value_per_100k_legs_inr": value / total_legs * 100_000 if total_legs else 0.0,
        })
    return pd.DataFrame(rows)


def annualise(value_inr: float, legs_days: float) -> float:
    """Scale a sampled-period figure to a year, stated explicitly rather than assumed."""
    if legs_days <= 0:
        return float("nan")
    return value_inr * (365.0 / legs_days)
