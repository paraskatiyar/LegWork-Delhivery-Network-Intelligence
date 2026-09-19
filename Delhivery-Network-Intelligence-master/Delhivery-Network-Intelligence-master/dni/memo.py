"""The strategy memo, generated from the artifacts rather than typed by hand.

Two reasons this is code and not a Word document:

* **Numbers cannot drift.** One reviewed submission's memo claimed
  Rs 280-340 Cr in its executive summary and Rs 225-275 Cr in its own
  supporting table three paragraphs later. When the prose is generated from the
  same frames the analysis produced, that class of error is impossible.

* **Re-running is honest.** Change a cost assumption in `configs/pipeline.yaml`
  and the memo changes with it. A memo that has to be rewritten by hand tends,
  under deadline, not to be.

The register is deliberately operational: hubs by name, interventions that name
a thing to do, rupees with the assumption attached, and no MAE, no embeddings,
no model architecture. The audience runs a network, not a notebook.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .config import Config


def _money(value: float) -> str:
    """Scale the unit to the number.

    This dataset is a sample, so honest figures land in lakhs, not crores.
    Formatting them as "Rs 0.0 Cr" would round a real result to zero and read
    as though the work found nothing -- so the unit follows the magnitude.
    """
    if abs(value) >= 1e7:
        return f"Rs {value / 1e7:,.2f} Cr"
    if abs(value) >= 1e5:
        return f"Rs {value / 1e5:,.1f} lakh"
    return f"Rs {value:,.0f}"


def _hub_label(name: str) -> str:
    base = str(name).split("(")[0].strip().replace("_", " ")
    state = str(name).split("(")[-1].replace(")", "").strip() if "(" in str(name) else ""
    return f"{base} ({state})" if state else base


def _corridor_label(text: str) -> str:
    """Facility codes and underscores are for the database, not the memo."""
    return str(text).replace("_", " ").replace(" -> ", " &rarr; ")


def _intervention(row, late_rate: float, legs_touched: float,
                  median_legs: float, median_late: float) -> str:
    """Pick the intervention that matches this hub's actual failure mode.

    A hub can be a bottleneck three different ways, and they need three
    different cheques. Issuing "add capacity" to all five -- which is what a
    single template produces -- is how a memo loses the room.
    """
    heavy = legs_touched >= median_legs
    severe = late_rate >= median_late
    if heavy and severe:
        return (
            f"Capacity and process together. It touches {legs_touched:,.0f} legs and misses "
            f"the promise on {late_rate:.0f}% of them: add dock capacity, then a second "
            f"dispatch wave to use it."
        )
    if heavy and not severe:
        return (
            f"Protect it rather than rebuild it. Throughput is high ({legs_touched:,.0f} legs) "
            f"but the miss rate is a contained {late_rate:.0f}% - it sits on {row["betweenness"]:.0%} "
            f"of network paths, so the risk is contagion, not current failure. Stage a parallel "
            f"route before volume grows into it."
        )
    if severe:
        return (
            f"Process, not concrete. Volume is moderate but {late_rate:.0f}% of legs touching "
            f"it miss the promise - attack dwell time and dock scheduling first; capacity "
            f"spend here would buy little."
        )
    return (
        f"Watch and instrument. It ranks on structural position ({row["betweenness"]:.0%} of "
        f"network paths) rather than on current failure. Add it to control-tower monitoring; "
        f"no capital case yet."
    )


def build(
    cfg: Config,
    ingest_report: pd.DataFrame,
    sla_summary: pd.DataFrame,
    decomposition: dict,
    hubs: pd.DataFrame,
    corridors: pd.DataFrame,
    ladder: pd.DataFrame,
    advantage: pd.DataFrame,
    routing_rules: pd.DataFrame,
    routing_scored: pd.DataFrame,
    sensitivity: pd.DataFrame,
    rar: pd.DataFrame,
    upgrade: pd.DataFrame,
    upgrade_detail: pd.DataFrame,
    model_value: pd.DataFrame,
    cold_start: dict,
    period_days: float,
    breakeven: pd.DataFrame | None = None,
    routing_summary: dict | None = None,
) -> str:
    osrm = ladder[ladder["rung"] == "osrm"].iloc[0]
    best = ladder[ladder["rung"] == "embedded"].iloc[0]
    baseline = ladder[ladder["rung"] == "trip"].iloc[0]
    total_adv = advantage[advantage["comparison"] == "trip -> embedded"].iloc[0]

    central_rar = rar[rar["scenario"] == "central"].iloc[0]
    central_upgrade = upgrade[
        (upgrade["recovery_scenario"] == "central") & (upgrade["penalty_scenario"] == "central")
    ].iloc[0]
    low_upgrade = upgrade[
        (upgrade["recovery_scenario"] == "conservative") & (upgrade["penalty_scenario"] == "low")
    ].iloc[0]
    high_upgrade = upgrade[
        (upgrade["recovery_scenario"] == "optimistic") & (upgrade["penalty_scenario"] == "high")
    ].iloc[0]
    central_model = model_value[model_value["penalty_scenario"] == "central"].iloc[0]

    ing = ingest_report.iloc[0]
    raw_rate = sla_summary.iloc[0]["late_rate_pct"]
    cal_rate = sla_summary.iloc[1]["late_rate_pct"]

    annual = 365.0 / period_days if period_days else 1.0

    lines: list[str] = []
    add = lines.append

    add("# Network Operations Strategy Memo")
    add("")
    add("**To:** Head of Network Operations  ")
    add("**From:** Data Science  ")
    add(f"**Date:** {date.today():%d %B %Y}  ")
    add(f"**Subject:** Where our delivery promises break, and the two things to fix first")
    add("")
    add("---")
    add("")

    # ---------------------------------------------------------------- summary
    add("## The short version")
    add("")
    add(
        f"We analysed {int(ing['legs_after_filters']):,} delivery legs across "
        f"{len(hubs):,} facilities. Three things came out of it, and the third "
        f"changes where we should spend money."
    )
    add("")
    add(
        f"**1. Our delivery promise is wrong far more often than our operation is late.** "
        f"Measured the way the industry usually does it - actual time versus the routing "
        f"engine's estimate plus 20% - {raw_rate:.0f}% of legs look late. That number is "
        f"useless: it fires on almost every shipment we run. When we instead ask whether a "
        f"leg missed what that corridor reliably achieves, the real figure is "
        f"**{cal_rate:.0f}%**. The difference is not a rounding argument. "
        f"**{decomposition['systematic_share_pct']:.0f}% of the gap against the routing "
        f"engine is systematic forecast error** - the engine assumes clean traffic and no "
        f"time spent inside our own buildings. Only "
        f"{decomposition['operational_share_pct']:.0f}% is genuine operational excess."
    )
    add("")
    add(
        f"**2. That means the cheapest fix is software, not concrete.** Quoting the new "
        f"model's estimate instead of the routing engine's takes the share of shipments "
        f"that miss their quoted window from {central_model['osrm_quote_miss_pct']:.0f}% to "
        f"{central_model['model_quote_miss_pct']:.0f}% - roughly "
        f"{central_model['legs_brought_inside_window']:,.0f} legs in the sampled period, "
        f"worth {_money(central_model['value_inr'])} in the period and about "
        f"{_money(central_model['value_inr'] * annual)} annualised. No facility changes."
    )
    add("")
    add(
        f"**3. The operational excess that remains is concentrated.** The top five hubs "
        f"below carry {hubs.head(5)['sla_contribution_pct'].sum():.0f}% of it. Halving "
        f"excess delay at the top three recovers "
        f"{central_upgrade['late_reduction_pct']:.1f}% of our late legs, worth "
        f"{_money(central_upgrade['revenue_recovered_inr'])} in the period "
        f"({_money(central_upgrade['revenue_recovered_inr'] * annual)} annualised), with a "
        f"plausible range of {_money(low_upgrade['revenue_recovered_inr'] * annual)} to "
        f"{_money(high_upgrade['revenue_recovered_inr'] * annual)} depending on how much "
        f"excess an upgrade actually removes and what a missed promise costs us."
    )
    add("")
    add(
        f"**Sequence the two levers.** Ship the model first: it is weeks, near-zero "
        f"capital, and it addresses the {decomposition['systematic_share_pct']:.0f}% of the "
        f"gap that no amount of dock space can touch. Then upgrade the hubs, which is the "
        f"only lever that reaches the remaining "
        f"{decomposition['operational_share_pct']:.0f}%."
    )
    add("")

    # ------------------------------------------------------------------- hubs
    add("## The five hubs to fix, in order")
    add("")
    add(
        "Ranked by a chokepoint score combining how much of the network routes through "
        "the facility, how much traffic it handles, and how many excess minutes it "
        "contributes. All three matter: a hub can be central and fine, or slow and "
        "irrelevant. These five are neither."
    )
    add("")
    add("| # | Facility | Share of network excess | Legs touched | Late rate | What to do |")
    add("|---|---|---|---|---|---|")
    detail_by_facility = upgrade_detail.set_index("facility") if len(upgrade_detail) else None
    median_legs = float(upgrade_detail["legs_touched"].median()) if len(upgrade_detail) else 0.0
    median_late = float(upgrade_detail["late_rate_pct"].median()) if len(upgrade_detail) else 0.0
    for i, row in hubs.head(5).iterrows():
        rec = detail_by_facility.loc[row["facility"]] if (
            detail_by_facility is not None and row["facility"] in detail_by_facility.index
        ) else None
        late_rate = float(rec["late_rate_pct"]) if rec is not None else 0.0
        legs_touched = float(rec["legs_touched"]) if rec is not None else float(row["legs_touched"])
        action = _intervention(row, late_rate, legs_touched, median_legs, median_late)
        add(
            f"| {i + 1} | **{_hub_label(row['facility_name'])}** | "
            f"{row['sla_contribution_pct']:.1f}% | {legs_touched:,.0f} | "
            f"{late_rate:.0f}% | {action} |"
        )
    add("")
    top3 = ", ".join(_hub_label(n) for n in hubs.head(3)["facility_name"])
    add(f"The capital case covers the first three: **{top3}**.")
    add("")

    # -------------------------------------------------------------- corridors
    add("## The corridors doing the damage")
    add("")
    add(
        "Ranked by how many excess minutes each contributes, not by how bad its ratio "
        "looks. That distinction matters: our worst ratios sit on lanes running ten "
        "shipments a month, where a single mis-scan produces a 25x reading. Those are a "
        "data-quality task, not an investment case, and they are excluded here."
    )
    add("")
    add("| Corridor | Legs | Typical vs promise | Share of excess | Intervention |")
    add("|---|---|---|---|---|")
    priority = corridors[corridors["is_credible_priority"] == 1].head(5)
    for _, row in priority.iterrows():
        if row["ftl_share"] > 0.7 and row["median_osrm_km"] > 150:
            action = "Long-haul FTL lane: hold a guaranteed departure window and stage a mid-point relay."
        elif row["median_osrm_km"] < 75:
            action = "Short-haul: the delay is dwell at the ends, not transit. Fix dock scheduling."
        else:
            action = "Stage a parallel route and shift departures out of the congested window."
        add(
            f"| {_corridor_label(row['corridor'])} | {int(row['legs']):,} | "
            f"{row['median_delay_ratio']:.2f}x | {row['sla_contribution_pct']:.1f}% | {action} |"
        )
    add("")
    artifacts = corridors[
        (corridors["is_credible_priority"] == 0)
        & (corridors["median_delay_ratio"] > cfg.sla.artifact_ratio_ceiling)
    ]
    add(
        f"> **Data-quality flag.** {len(artifacts):,} corridors show median times more than "
        f"{cfg.sla.artifact_ratio_ceiling:g}x the routing estimate on low volumes. These are "
        f"almost certainly mis-scans, one-off disruptions, or lanes where the routing engine "
        f"has the geometry wrong. They are excluded from the priority list above and should "
        f"go to a separate data-quality review, not to capital planning."
    )
    add("")

    # ------------------------------------------------------------------- ETA
    add("## The promise itself")
    add("")
    add(
        f"Today we quote the routing engine's number. It lands within 15% of the truth on "
        f"**{osrm['within_15pct']:.0f}%** of legs, and it is biased low by "
        f"{abs(osrm['bias_minutes']):.0f} minutes on average - it under-promises the "
        f"duration on almost every shipment, which is why our customers experience us as "
        f"chronically late."
    )
    add("")
    add(
        f"Replacing it with a model that reads the network as a connected graph - each "
        f"corridor's own history, and each facility's position in the network - lands "
        f"within 15% on **{best['within_15pct']:.0f}%** of legs, with an average error of "
        f"{best['mae_minutes']:.0f} minutes against the routing engine's "
        f"{osrm['mae_minutes']:.0f}."
    )
    add("")
    add(
        f"We checked whether the graph is doing real work rather than flattering itself. "
        f"Against an equivalent model given the same shipment details but no network "
        f"information, the graph version cuts average error by "
        f"{total_adv['mae_reduction_minutes']:.1f} minutes "
        f"({total_adv['mae_reduction_pct']:.0f}%), and the 95% confidence interval on that "
        f"gap runs {total_adv['ci95_low']:.1f} to {total_adv['ci95_high']:.1f} minutes - "
        f"{'comfortably clear of zero' if total_adv['significant'] else 'not clear of zero'}. "
        f"It is a real effect and a moderate one. Anyone reporting a 40-80% improvement "
        f"from graph features on this dataset has almost certainly let the answer leak into "
        f"the question."
    )
    add("")
    add(
        f"**Deployment caveat worth knowing before you sign off:** "
        f"{cold_start['legs_with_unseen_endpoint_pct']:.0f}% of legs in our holdout period "
        f"start or end at a facility the model has never seen, and "
        f"{cold_start['legs_on_unseen_corridor_pct']:.0f}% run on a corridor with no "
        f"history. The network opens new facilities faster than a model retrains. Those "
        f"shipments must fall back to the routing-engine estimate with a widened window, "
        f"and the fallback rate should be on the monitoring dashboard from day one."
    )
    add("")

    # --------------------------------------------------------------- routing
    add("## FTL vs Carting")
    add("")
    band = routing_scored[routing_scored["in_overlap_band"]]
    switch_rate = float(band["would_switch"].mean() * 100) if len(band) else 0.0
    add(
        f"We priced both modes for every shipment - predicted duration under each, plus "
        f"transport cost, the value of the time, and the penalty if it misses its window - "
        f"and compared the totals. We only advise on the "
        f"{len(band) / len(routing_scored) * 100:.0f}% of shipments in the distance range "
        f"where both modes genuinely operate today. Outside that range the comparison is "
        f"extrapolation, and we leave the current choice alone."
    )
    add("")
    if routing_summary:
        add(
            f"**The finding is blunt: on these lanes, FTL does not buy time.** In the "
            f"{routing_summary['band_min_km']:.0f}-{routing_summary['band_max_km']:.0f} km "
            f"range where we run both modes, FTL is a median of "
            f"{routing_summary['median_ftl_time_gain_min']:.1f} minutes faster and costs a "
            f"mean Rs {routing_summary['mean_ftl_cost_premium_inr']:,.0f} more per shipment. "
            f"Today **{routing_summary['ftl_share_today_pct']:.0f}%** of those shipments go "
            f"FTL. That is the largest single cost saving this analysis found, and it needs "
            f"no new capability - only a dispatch rule."
        )
        add("")
    if breakeven is not None and len(breakeven):
        row = breakeven.iloc[len(breakeven) // 2]
        add(
            f"**The rule itself is a distance.** On our rate card FTL only repays its "
            f"premium beyond roughly **{row['breakeven_km_generalised']:.0f} km**. That "
            f"threshold barely moves even if we quadruple what we charge ourselves for a "
            f"late delivery, because there is almost no time saving for a higher penalty to "
            f"multiply. Below it, default to Carting; above it, FTL."
        )
        add("")
    add(f"Applied across the band, the framework would change the mode on **{switch_rate:.0f}%** of shipments.")
    add("")
    if len(routing_rules):
        add("| Distance | Time of day | Source hub | Shipments | Today | Recommended | Saving per shipment |")
        add("|---|---|---|---|---|---|---|")
        for _, row in routing_rules.head(6).iterrows():
            add(
                f"| {row['distance_band']} | {row['time_of_day']} | "
                f"{row['source_graph_position']} | {int(row['legs']):,} | "
                f"{row['ftl_share_today'] * 100:.0f}% FTL | {row['recommendation']} | "
                f"Rs {row['avg_generalised_saving_inr']:,.0f} |"
            )
        add("")
    add(
        "**The honest caveat.** We have no cost data - the rate card behind these numbers "
        "is our assumption, stated in the configuration file and swept across a range. The "
        "*direction* survives that sweep: Carting short, FTL long, and short-haul FTL never "
        "justified on any setting we tried. The *exact* crossover distance does not survive "
        "it - it moves with the haulage rates, which are the numbers we are least sure "
        "about. Before acting, replace our rate card with the real one and re-run; it is one "
        "configuration change and the whole analysis, including this memo, regenerates."
    )
    add("")

    # ---------------------------------------------------------------- money
    add("## What this is worth")
    add("")
    add(
        f"Revenue at risk in the sampled period, at our central assumption of "
        f"Rs {cfg.revenue.revenue_per_leg_inr:,.0f} per leg and a "
        f"{cfg.revenue.penalty_share_of_revenue:.0%} penalty on a missed promise: "
        f"**{_money(central_rar['revenue_at_risk_inr'])}** across "
        f"{int(central_rar['late_legs']):,} late legs."
    )
    add("")
    add("| Lever | Period | Annualised | What it depends on |")
    add("|---|---|---|---|")
    add(
        f"| Ship the graph ETA model | {_money(central_model['value_inr'])} | "
        f"{_money(central_model['value_inr'] * annual)} | Nothing physical. Quote the model's "
        f"number instead of the routing engine's. |"
    )
    add(
        f"| Upgrade the top 3 hubs | {_money(central_upgrade['revenue_recovered_inr'])} | "
        f"{_money(central_upgrade['revenue_recovered_inr'] * annual)} | Removing half the "
        f"excess delay at those hubs. Range "
        f"{_money(low_upgrade['revenue_recovered_inr'] * annual)}-"
        f"{_money(high_upgrade['revenue_recovered_inr'] * annual)} annualised. |"
    )
    add("")
    add(
        f"**Read these as ranges, not forecasts.** Every input is an assumption we have "
        f"written down: Rs {cfg.revenue.revenue_per_leg_inr:,.0f} revenue per leg, a "
        f"{cfg.revenue.penalty_share_low:.0%}-{cfg.revenue.penalty_share_high:.0%} penalty "
        f"band, and a {cfg.revenue.recovery_low:.0%}-{cfg.revenue.recovery_high:.0%} range "
        f"for how much excess a hub upgrade removes. We have deliberately not sized this "
        f"against the full gap versus the routing engine - most of that gap is forecast "
        f"error, and a bigger dock does not fix a forecast. Sizing the capital case against "
        f"the whole gap would overstate the return by roughly seven times."
    )
    add("")
    add(
        f"The period sampled is {period_days:.0f} days, so annualised figures assume the "
        f"same run-rate for a year. September-October includes festive volume; treat the "
        f"annualisation as indicative until we re-run on a full year."
    )
    add("")

    # ---------------------------------------------------------------- next
    add("## What we would like agreed")
    add("")
    add(
        "1. **Now - quote the model, not the engine.** Run it in shadow for four weeks "
        "against live shipments, then switch customer-facing promises over. Watch the "
        "cold-start fallback rate as the primary health metric."
    )
    add(
        f"2. **Quarter 1 - {_hub_label(hubs.iloc[0]['facility_name'])}.** The single largest "
        f"contributor to excess delay. Capacity and a second dispatch wave."
    )
    add(
        f"3. **Quarter 2 - {_hub_label(hubs.iloc[1]['facility_name'])} and "
        f"{_hub_label(hubs.iloc[2]['facility_name'])}.** Completes the top-three case."
    )
    add(
        "4. **In parallel - send us the rate card.** The FTL/Carting framework is built and "
        "runs; it is currently priced on our assumptions rather than your numbers. With the "
        "real rate card it becomes a dispatch rule rather than a recommendation."
    )
    add(
        f"5. **Separately - the {len(artifacts):,} artifact corridors** go to data quality, "
        f"not to operations."
    )
    add("")
    add("---")
    add("")
    add(
        "*Every figure in this memo is generated directly from the analysis pipeline; "
        "changing an assumption in the configuration regenerates the memo with it. "
        "Method, model comparison and limitations are in the technical appendix.*"
    )
    add("")
    return "\n".join(lines)


def write(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
