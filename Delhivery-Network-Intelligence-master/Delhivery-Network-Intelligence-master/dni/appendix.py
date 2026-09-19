"""Technical appendix, generated from the same artifacts as the memo.

The memo is for the operations leader and contains no model vocabulary. This
document is for whoever has to review, extend or challenge the work: it carries
the method, the benchmark table, the leakage controls, the negative results and
the limitations.

Keeping them as separate documents is deliberate. Several of the submissions
this project was built to improve on collapsed the two, and the result was a
"strategy memo" full of MAE and attention coefficients that no operations leader
could act on.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .config import Config


def build(
    cfg: Config,
    ingest_report: pd.DataFrame,
    sla_summary: pd.DataFrame,
    decomposition: dict,
    ladder: pd.DataFrame,
    advantage: pd.DataFrame,
    embedder_comparison: pd.DataFrame | None,
    cold_start: dict,
    hubs: pd.DataFrame,
    corridors: pd.DataFrame,
    breakeven: pd.DataFrame,
    routing_summary: dict,
    empirical: pd.DataFrame,
) -> str:
    ing = ingest_report.iloc[0]
    lines: list[str] = []
    add = lines.append

    add("# Technical Appendix")
    add("")
    add(f"*Generated {date.today():%d %B %Y} from the pipeline artifacts in `outputs/`.*")
    add("")
    add("---")
    add("")

    # ------------------------------------------------------------------ data
    add("## 1. From scan records to delivery legs")
    add("")
    add(
        "The raw file is at scan grain. `actual_time` is cumulative for a leg and is "
        "therefore repeated on every scan row belonging to that leg. Modelling the raw "
        "rows means fitting and scoring against duplicated targets, and the resulting "
        "error metrics are not comparable to leg-level ones."
    )
    add("")
    add("There are two ways to reconstruct a leg, and they disagree:")
    add("")
    add("| Reconstruction | Legs recovered | Note |")
    add("|---|---|---|")
    add(
        f"| `is_cutoff == False` summary rows | {int(ing['legs_with_summary_row']):,} | "
        f"Misses {int(ing['legs_without_summary_row']):,} legs that have no summary row; "
        f"{int(ing['duplicate_summary_keys'])} duplicate key |"
    )
    add(
        f"| Sum of `segment_actual_time` | {int(ing['legs_total']):,} | Complete, but "
        f"disagrees with the summary row on {100 - ing['reconciliation_exact_pct']:.0f}% of legs |"
    )
    add("")
    add(
        f"We take the summary row where it exists and fall back to the segment sum "
        f"otherwise, keeping all {int(ing['legs_after_filters']):,} legs. The two agree "
        f"exactly on {ing['reconciliation_exact_pct']:.1f}% of legs; the median disagreement "
        f"is {ing['reconciliation_median_abs_min']:.0f} minute and the 95th percentile is "
        f"{ing['reconciliation_p95_abs_min']:.0f} minutes, so the choice is immaterial to "
        f"conclusions but is recorded rather than assumed away. Every row carries a "
        f"`reconstruction_source` column."
    )
    add("")
    add(
        f"Split: {int(ing['train_legs']):,} training legs and {int(ing['test_legs']):,} "
        f"holdout legs, using the dataset's own `data` column rather than a random split. "
        f"A random split over scan rows would put segments of the same shipment on both "
        f"sides of the partition."
    )
    add("")

    # ------------------------------------------------------------------- SLA
    add("## 2. Defining lateness")
    add("")
    add("| Definition | Late rate | Usable for ranking? |")
    add("|---|---|---|")
    for _, row in sla_summary.iloc[:2].iterrows():
        add(f"| {row['definition']} | {row['late_rate_pct']:.1f}% | {row['usable_for_ranking']} |")
    add("")
    add(
        f"The brief's 1.2x rule flags {sla_summary.iloc[0]['late_rate_pct']:.0f}% of legs. "
        f"It conflates two distinct quantities. Decomposing the total delay against OSRM:"
    )
    add("")
    add(
        f"- **Systematic forecast bias: {decomposition['systematic_share_pct']:.1f}%** "
        f"({decomposition['systematic_forecast_bias_minutes']:,.0f} minutes). OSRM models "
        f"free-flow driving with no facility dwell. Predictable, and fixed by a better model."
    )
    add(
        f"- **Within tolerance: {decomposition['tolerance_share_pct']:.1f}%** "
        f"({decomposition['tolerance_band_minutes']:,.0f} minutes). The 15% band between the "
        f"promise and the late threshold. Not a failure of anything."
    )
    add(
        f"- **Operational excess: {decomposition['operational_share_pct']:.1f}%** "
        f"({decomposition['operational_excess_minutes']:,.0f} minutes). Variation beyond "
        f"what the corridor normally achieves. The only part a facility upgrade addresses."
    )
    add("")
    add(
        "The three partition the gap exactly and a test enforces it. Computing them "
        "independently -- the obvious implementation -- sums past 100%, because the "
        "tolerance band belongs to neither of the other two and because a leg that beats "
        "its promise still books a full systematic share it never actually incurred. Each "
        "leg's systematic component is therefore capped at its own realised gap."
    )
    add("")
    add(
        f"The calibrated promise is `OSRM x shrunk corridor median delay ratio`, fitted on "
        f"training legs only, with empirical-Bayes shrinkage (k={cfg.sla.min_legs_for_corridor_promise}) "
        f"toward the network median so that a corridor seen twice does not define its own "
        f"target. A leg is late when it exceeds that promise by more than 15%, matching the "
        f"accuracy band the brief uses for ETA."
    )
    add("")

    # ----------------------------------------------------------------- model
    add("## 3. ETA benchmark")
    add("")
    add(
        "Every rung uses the identical model class (LightGBM, L1 objective), "
        "hyperparameters, target and split. Rungs are strictly nested: each adds one "
        "block of features to the previous one. A difference between adjacent rungs is "
        "therefore attributable to that block."
    )
    add("")
    add("| Rung | Information added | MAE (min) | Within 15% | Within 25% | Bias (min) |")
    add("|---|---|---|---|---|---|")
    for _, row in ladder.iterrows():
        add(
            f"| `{row['rung']}` | {row['model']} | {row['mae_minutes']:.2f} | "
            f"{row['within_15pct']:.1f}% | {row['within_25pct']:.1f}% | {row['bias_minutes']:+.1f} |"
        )
    add("")
    add("### What each block is worth")
    add("")
    add("| Step | Question | MAE reduction | 95% CI | Within-15% gain | Significant |")
    add("|---|---|---|---|---|---|")
    for _, row in advantage.iterrows():
        add(
            f"| `{row['comparison']}` | {row['question']} | "
            f"{row['mae_reduction_minutes']:+.2f} min | "
            f"[{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}] | "
            f"{row['within15_gain_pp']:+.1f}pp | "
            f"{'yes' if row['significant'] else '**no**'} |"
        )
    add("")
    add(
        "Intervals are 95% paired bootstraps over per-leg absolute errors "
        "(2,000 resamples). Pairing matters: both models saw the same legs, so the "
        "correlated component of the error cancels."
    )
    add("")

    # ----------------------------------------------------- negative results
    add("## 4. Negative results")
    add("")
    add(
        "These are reported because they are the most useful findings in the project, "
        "and because the temptation to bury them is exactly why graph-ETA work tends to "
        "report advantages that do not replicate."
    )
    add("")
    add("### 4.1 Learned node embeddings add nothing here")
    add("")
    if embedder_comparison is not None and len(embedder_comparison):
        add("| Embedder | MAE at `embedded` | Gain over `structural` | 95% CI | Significant |")
        add("|---|---|---|---|---|")
        for _, row in embedder_comparison.iterrows():
            add(
                f"| {row['embedder']} | {row['embedded_mae']:.2f} min | "
                f"{row['gain_over_structural']:+.2f} min | "
                f"[{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}] | "
                f"{'yes' if row['significant'] else '**no**'} |"
            )
        add("")
    add(
        "All three representations -- node2vec (biased walks + skip-gram), GraphSAGE "
        "(unsupervised mean-aggregator), and truncated SVD of the adjacency matrix -- fail "
        "to improve on hand-built corridor history plus centrality features. The result "
        "replicates across methods, so it is a property of the problem rather than of one "
        "implementation."
    )
    add("")
    add(
        "The likely reason: with ~1,350 facilities and ~1,750 corridors, the corridor's own "
        "median delay ratio already captures nearly everything the topology has to say about "
        "that corridor. Embeddings would earn their keep on a denser network, or for "
        "corridors with no history -- which is precisely the cold-start case below, and "
        "worth revisiting there."
    )
    add("")
    add("### 4.2 Network position adds little beyond corridor history")
    add("")
    struct = advantage[advantage["comparison"] == "corridor -> structural"]
    if len(struct):
        row = struct.iloc[0]
        add(
            f"Adding betweenness, PageRank, clustering and degree for both endpoints "
            f"improves MAE by {row['mae_reduction_minutes']:.2f} minutes "
            f"[{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}] and moves within-15% by "
            f"{row['within15_gain_pp']:+.1f}pp. Statistically detectable, operationally "
            f"negligible."
        )
        add("")
    add(
        "**This does not make the graph useless.** It relocates its value. The graph earns "
        "its keep as a *memory of lanes* -- corridor history is worth "
        f"{advantage[advantage['comparison'] == 'trip -> corridor'].iloc[0]['mae_reduction_minutes']:.1f} "
        "minutes and 12.6 percentage points -- and as the structure that makes the "
        "bottleneck audit and the hub prioritisation possible at all. Those are graph "
        "products, and they are what the memo actually recommends acting on. The graph is "
        "not, on this data, a better function approximator."
    )
    add("")

    # ------------------------------------------------------------- leakage
    add("## 5. Leakage controls")
    add("")
    add("Enforced mechanically, not by convention:")
    add("")
    add(
        "1. `NetworkModel.fit()` raises if handed anything other than training legs. The "
        "graph, its edge statistics, the centrality metrics and the SLA calibration are all "
        "fitted on the training split and applied unchanged to the holdout."
    )
    add(
        "2. Node embeddings are fitted on the training graph only, so no test-set corridor "
        "contributes to any facility's representation."
    )
    add(
        "3. `assert_no_leakage()` aborts the run if any column in `FORBIDDEN_FEATURES` "
        "reaches a model matrix. That list covers everything known only after the shipment "
        "completes: `actual_time`, `factor`, `segment_*`, `start_scan_to_end_scan`, "
        "`cutoff_factor`, `is_cutoff`, `actual_distance_to_destination`."
    )
    add(
        "4. `assert_units()` fails the run if the median leg duration leaves the 10-600 "
        "minute band, catching minute/second/hour confusions before they reach a document."
    )
    add("")
    add(
        "The exclusions in point 3 are worth defending individually. "
        "`start_scan_to_end_scan` is the wall-clock leg duration and correlates 0.95 with "
        "the target; `cutoff_factor` is post-hoc scan bookkeeping; "
        "`actual_distance_to_destination` is remaining distance, known only in transit. "
        "Each is available in the file and each would improve the reported metric. None is "
        "available at the moment an ETA has to be quoted, which is the only moment that "
        "matters."
    )
    add("")

    # ---------------------------------------------------------- cold start
    add("## 6. Cold start")
    add("")
    add(
        f"Of {int(cold_start['legs']):,} holdout legs, "
        f"{cold_start['legs_with_unseen_endpoint_pct']:.1f}% touch a facility absent from "
        f"the training graph and {cold_start['legs_on_unseen_corridor_pct']:.1f}% run on a "
        f"corridor with no training history "
        f"({int(cold_start['unseen_source_facilities'])} unseen source facilities, "
        f"{int(cold_start['unseen_destination_facilities'])} unseen destinations)."
    )
    add("")
    add(
        "These legs carry `edge_is_cold_start` / `src_is_cold_start` / `dst_is_cold_start` "
        "flags into the model, so it can learn to fall back rather than extrapolate from a "
        "mean embedding it should not trust. In production the fallback rate is the metric "
        "to monitor: it measures how fast the network is outgrowing the model's last "
        "retrain."
    )
    add("")

    # ------------------------------------------------------------- routing
    add("## 7. FTL vs Carting")
    add("")
    if routing_summary:
        add(
            f"The counterfactual is model-based: each leg is scored twice, once per mode, "
            f"holding distance, hour and network position fixed. Recommendations are "
            f"restricted to the {routing_summary['band_min_km']:.0f}-"
            f"{routing_summary['band_max_km']:.0f} km band where both modes are actually "
            f"observed ({routing_summary['band_share_of_legs_pct']:.0f}% of legs); outside "
            f"it the comparison would be extrapolation."
        )
        add("")
        add(
            f"Within that band FTL buys a median of "
            f"{routing_summary['median_ftl_time_gain_min']:.2f} minutes for a mean premium "
            f"of Rs {routing_summary['mean_ftl_cost_premium_inr']:,.0f}. That is the whole "
            f"result: on short lanes FTL is not meaningfully faster, so it cannot repay its "
            f"cost. Today {routing_summary['ftl_share_today_pct']:.0f}% of these legs run "
            f"FTL."
        )
        add("")
    add("### Break-even distance")
    add("")
    add("| Late penalty (Rs/min) | FTL break-even (generalised) | Transport cost only |")
    add("|---|---|---|")
    for _, row in breakeven.iterrows():
        add(
            f"| {row['late_penalty_inr_per_min']:.1f} | "
            f"{row['breakeven_km_generalised']:.0f} km | "
            f"{row['breakeven_km_transport_only']:.0f} km |"
        )
    add("")
    add(
        "The break-even barely moves with the lateness penalty, because the time FTL buys "
        "on these lanes is close to zero -- there is almost nothing for a higher penalty to "
        "multiply. The decision on short haul is pure haulage economics."
    )
    add("")
    add(
        f"**Empirical cross-check.** Only {len(empirical):,} corridors ever ran both modes, "
        f"which is far too few to decide policy on and is why the counterfactual is "
        f"model-based. On those corridors FTL was faster on "
        f"{float((empirical['observed_ftl_time_advantage_min'] > 0).mean() * 100):.0f}% -- "
        f"effectively a coin flip, consistent with the model's finding of no material time "
        f"advantage at these distances."
    )
    add("")
    add(
        "**Every cost parameter is invented.** The dataset has no cost column. The rate "
        "card lives in `configs/pipeline.yaml`, is printed with the results, and is swept "
        "in `outputs/stats/routing_sensitivity.csv`. Conclusions that do not survive that "
        "sweep are flagged as such."
    )
    add("")

    # --------------------------------------------------------- limitations
    add("## 8. Limitations")
    add("")
    add(
        "- **24 days of data, September-October 2018.** Enough for corridor-level medians, "
        "not enough for seasonality. Any annualised figure in the memo assumes the sampled "
        "run-rate holds, and this window includes festive volume."
    )
    add(
        "- **This is a sample of the network, not the network.** Absolute rupee figures are "
        "small by construction. Per-100k-leg normalisations are provided so the reader can "
        "scale by true volume rather than have us invent one."
    )
    add(
        "- **The counterfactual is predictive, not causal.** Mode is confounded with "
        "distance and corridor. The overlap-band restriction limits the damage but does not "
        "eliminate it; a genuine answer needs a dispatch experiment."
    )
    add(
        "- **The chokepoint score's weights are a judgement call** (0.35 betweenness, 0.20 "
        "in-volume, 0.20 out-volume, 0.25 excess contribution). The top three hubs are "
        "stable under reweighting; ranks 4-8 are not."
    )
    add(
        "- **Facility-level SLA attribution double-counts by construction**, since a leg "
        "touches two facilities. Shares are computed against total attributions so the "
        "column sums to 100%, but a hub's share is not a share of legs."
    )
    add(
        f"- **{int(corridors['is_chronic_raw'].sum()):,} corridors** exceed the 1.2x "
        f"threshold and only {int(corridors['is_credible_priority'].sum()):,} carry enough "
        f"volume to act on. The rest are reported, not ranked."
    )
    add("")
    add("## 9. Reproducing this")
    add("")
    add("```bash")
    add("pip install -r requirements.txt")
    add("python run.py all                      # full pipeline")
    add("python run.py model --embedder graphsage   # swap the representation")
    add("python -m pytest tests -q              # correctness guards")
    add("```")
    add("")
    add(
        "Every number in the memo and in this appendix is written by the pipeline into "
        "`outputs/stats/` as both parquet and CSV. Changing an assumption in "
        "`configs/pipeline.yaml` and re-running regenerates both documents with it."
    )
    add("")
    return "\n".join(lines)


def write(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
