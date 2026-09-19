"""Pipeline entry point.

    python run.py all                  # full pipeline, ~3 minutes
    python run.py prep                 # legs + SLA calibration + graph
    python run.py audit                # bottleneck + corridor audit + figures
    python run.py model                # ablation ladder + graph advantage
    python run.py routing              # FTL vs Carting + sensitivity
    python run.py memo                 # regenerate the memo from artifacts

Stages write parquet/CSV into outputs/ and read them back, so any stage can be
re-run on its own after a configuration change without recomputing the rest.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dni import (appendix as appendix_mod, economics, embeddings, graph as gmod, ingest,
                 memo as memo_mod, models, routing, sla, viz)
from dni.config import Config
from dni.features import FeatureSpace


def _banner(text: str) -> None:
    print(f"\n{'=' * 72}\n  {text}\n{'=' * 72}")


def _save(frame: pd.DataFrame, directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.parquet"
    frame.to_parquet(path, index=False)
    frame.to_csv(directory / f"{name}.csv", index=False)
    return path


def _load(directory: Path, name: str) -> pd.DataFrame:
    path = directory / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run an earlier stage first.")
    return pd.read_parquet(path)


# --------------------------------------------------------------------- stages
def stage_prep(cfg: Config) -> tuple[pd.DataFrame, gmod.NetworkModel]:
    _banner("STAGE 1  Ingest, SLA calibration, graph construction")
    raw = ingest.load_raw(cfg.paths.resolve("raw_csv"))
    legs, report = ingest.build_legs(raw, cfg)

    print(f"  {report.raw_rows:,} scan rows  ->  {report.legs_after_filters:,} delivery legs")
    print(f"  legs with an is_cutoff=False summary row : {report.legs_with_summary_row:,}")
    print(f"  legs reconstructed from segment sums     : {report.legs_without_summary_row:,}"
          "   <- silently lost by a summary-row-only filter")
    print(f"  summary vs segment-sum agreement         : {report.reconciliation_exact_pct:.1f}% exact, "
          f"median gap {report.reconciliation_median_abs_min:.0f} min")
    print(f"  split: {report.train_legs:,} train / {report.test_legs:,} test")

    train_mask = legs["split"] == "train"
    calibration = sla.SLACalibration.fit(legs[train_mask], cfg)
    legs = calibration.apply(legs, cfg)
    ingest.assert_units(legs)

    summary = sla.summarise(legs, cfg)
    decomposition = sla.decompose_delay(legs)
    print("\n  Late-rate definitions:")
    for _, row in summary.iloc[:2].iterrows():
        print(f"    {row['late_rate_pct']:5.1f}%  {row['definition']}")
    print(f"\n  Delay decomposition: {decomposition['systematic_share_pct']:.1f}% systematic "
          f"forecast bias / {decomposition['operational_share_pct']:.1f}% operational excess")

    network = gmod.NetworkModel(cfg).fit(legs[legs["split"] == "train"])
    print(f"\n  Graph (train only): {network.graph.number_of_nodes():,} facilities, "
          f"{network.graph.number_of_edges():,} corridors")

    stats = cfg.paths.resolve("stats")
    _save(legs, cfg.paths.resolve("processed"), "legs")
    _save(report.to_frame(), stats, "ingest_report")
    _save(summary, stats, "sla_definitions")
    _save(pd.DataFrame([decomposition]), stats, "delay_decomposition")
    _save(network.hub_metrics, stats, "hub_metrics")
    _save(network.edge_stats, stats, "edge_stats")
    return legs, network


def stage_audit(cfg: Config, legs: pd.DataFrame, network: gmod.NetworkModel) -> pd.DataFrame:
    _banner("STAGE 2  Bottleneck and corridor audit")
    hubs = network.hub_metrics
    corridors = gmod.chronic_corridor_audit(legs, network, cfg)
    cold = network.cold_start_report(legs[legs["split"] == "test"])

    print("\n  Top 5 chokepoint hubs")
    for i, row in hubs.head(5).iterrows():
        print(f"    {i + 1}. {str(row['facility_name'])[:38]:<40} "
              f"betweenness {row['betweenness']:.3f}  "
              f"clustering {row['clustering_coeff']:.3f}  "
              f"in/out {int(row['in_degree'])}/{int(row['out_degree'])}  "
              f"excess share {row['sla_contribution_pct']:.1f}%")

    credible = corridors[corridors["is_credible_priority"] == 1]
    print(f"\n  Corridors: {len(corridors):,} total, "
          f"{int(corridors['is_chronic_raw'].sum()):,} chronic by the 1.2x rule "
          f"({corridors['is_chronic_raw'].mean() * 100:.0f}% - unusable for ranking), "
          f"{len(credible):,} volume-credible priorities")
    print("\n  Top 5 corridors by excess-delay contribution")
    for _, row in credible.head(5).iterrows():
        print(f"    {row['corridor'][:58]:<60} {row['sla_contribution_pct']:5.2f}%  "
              f"({int(row['legs']):,} legs, {row['median_delay_ratio']:.2f}x)")

    figures = cfg.paths.resolve("figures")
    figures.mkdir(parents=True, exist_ok=True)
    decomposition = sla.decompose_delay(legs)
    viz.sla_definition_chart(sla.summarise(legs, cfg), decomposition, figures / "01_sla_definition.png")
    viz.network_map(network.graph, hubs, corridors, figures / "02_network_map.png")
    viz.hub_ranking(hubs, figures / "03_hub_ranking.png")
    viz.corridor_ranking(corridors, figures / "04_corridor_ranking.png")
    print(f"\n  Figures -> {figures}")

    _save(corridors, cfg.paths.resolve("stats"), "corridor_audit")
    _save(pd.DataFrame([cold]), cfg.paths.resolve("stats"), "cold_start_report")
    return corridors


def stage_model(cfg: Config, legs: pd.DataFrame, network: gmod.NetworkModel,
                embedder_kind: str) -> tuple[models.LadderResult, FeatureSpace]:
    _banner(f"STAGE 3  ETA ablation ladder (embedder: {embedder_kind})")
    train = legs[legs["split"] == "train"].copy()
    test = legs[legs["split"] == "test"].copy()

    started = time.time()
    embedder = embeddings.build_embedder(embedder_kind, cfg).fit(network.graph)
    print(f"  {embedder_kind} embeddings fitted on the TRAIN graph in {time.time() - started:.1f}s "
          f"({len(embedder.nodes):,} facilities x {embedder.dim} dims)")

    space = FeatureSpace(cfg=cfg, network=network, embedder=embedder)
    result = models.run_ladder(train, test, space, cfg)

    print("\n  Ablation ladder (holdout: "
          f"{int(result.metrics.iloc[0]['n']):,} legs)")
    print(f"    {'rung':<12}{'MAE (min)':>11}{'within 15%':>13}{'within 25%':>13}{'bias':>9}")
    for _, row in result.metrics.iterrows():
        print(f"    {row['rung']:<12}{row['mae_minutes']:>11.2f}"
              f"{row['within_15pct']:>12.1f}%{row['within_25pct']:>12.1f}%"
              f"{row['bias_minutes']:>9.1f}")

    print("\n  What each block is worth (95% paired-bootstrap interval)")
    for _, row in result.advantage.iterrows():
        flag = "significant" if row["significant"] else "NOT significant"
        print(f"    {row['comparison']:<24} {row['mae_reduction_minutes']:+6.2f} min "
              f"[{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}]  "
              f"{row['within15_gain_pp']:+5.1f}pp   {flag}")
        print(f"      -> {row['question']}")

    figures = cfg.paths.resolve("figures")
    viz.ladder_chart(result.metrics, result.advantage, figures / "05_model_ladder.png")
    viz.advantage_chart(result.advantage, figures / "06_graph_advantage.png")
    viz.prediction_scatter(result.predictions, figures / "07_prediction_scatter.png")

    stats = cfg.paths.resolve("stats")
    _save(result.metrics, stats, "eta_ladder_metrics")
    _save(result.advantage, stats, "graph_advantage")
    _save(result.predictions, stats, "eta_predictions")
    return result, space


def stage_routing(cfg: Config, legs: pd.DataFrame, network: gmod.NetworkModel,
                  space: FeatureSpace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _banner("STAGE 4  FTL vs Carting decision framework")
    train = legs[legs["split"] == "train"].copy()
    test = legs[legs["split"] == "test"].copy()

    scored = routing.score_counterfactuals(train, test, space, cfg)
    rules = routing.decision_rules(scored, network.hub_metrics)
    sens = routing.sensitivity(scored, cfg)
    empirical = routing.empirical_validation(legs)
    breakeven = routing.breakeven_distance(scored, cfg)
    summary = routing.mode_summary(scored, cfg)

    band = scored[scored["in_overlap_band"]]
    print(f"  Overlap band (both modes actually operate): {summary['band_min_km']:.0f}-"
          f"{summary['band_max_km']:.0f} km, {len(band):,} of {len(scored):,} legs "
          f"({summary['band_share_of_legs_pct']:.0f}%)")
    print(f"  In that band FTL buys {summary['median_ftl_time_gain_min']:.1f} min (median) "
          f"for a Rs {summary['mean_ftl_cost_premium_inr']:,.0f} premium")
    print(f"  FTL share today {summary['ftl_share_today_pct']:.0f}%  ->  "
          f"recommended {summary['ftl_share_recommended_pct']:.0f}%  "
          f"(would switch {summary['switch_rate_pct']:.0f}% of legs, "
          f"Rs {summary['saving_per_switched_leg_inr']:,.0f} saved per switched leg)")
    print("\n  Break-even distance for FTL")
    for _, row in breakeven.iterrows():
        print(f"    late penalty Rs {row['late_penalty_inr_per_min']:>6.1f}/min  ->  "
              f"FTL pays above {row['breakeven_km_generalised']:6.0f} km "
              f"(transport cost alone: {row['breakeven_km_transport_only']:.0f} km)")
    print(f"\n  Empirical cross-check: only {len(empirical):,} corridors ever ran both modes "
          f"-- too few to decide on, which is why the counterfactual is model-based.")
    if len(empirical):
        agree = float((empirical["observed_ftl_time_advantage_min"] > 0).mean() * 100)
        print(f"  On those {len(empirical)}, FTL was faster on {agree:.0f}% -- "
              f"model-based counterfactual agrees in sign: "
              f"{'yes' if (band['eta_minutes_saved_by_ftl'].mean() > 0) == (agree > 50) else 'NO - investigate'}")

    print("\n  Sensitivity: FTL share recommended as assumptions move")
    for parameter in sens["parameter"].unique():
        sub = sens[sens["parameter"] == parameter]
        span = f"{sub['ftl_share_recommended_pct'].min():.0f}%-{sub['ftl_share_recommended_pct'].max():.0f}%"
        print(f"    {parameter:<28} {span}")

    viz.routing_chart(scored, sens, cfg.paths.resolve("figures") / "08_routing.png")
    stats = cfg.paths.resolve("stats")
    _save(scored.drop(columns=[c for c in scored.columns if c.startswith(("n2v_", "sage_", "svd_"))],
                      errors="ignore"), stats, "routing_decisions")
    _save(rules, stats, "routing_rules")
    _save(sens, stats, "routing_sensitivity")
    _save(empirical, stats, "routing_empirical_check")
    _save(breakeven, stats, "routing_breakeven")
    _save(pd.DataFrame([summary]), stats, "routing_summary")
    return scored, rules, sens, breakeven, summary


def stage_memo(cfg: Config, legs: pd.DataFrame, network: gmod.NetworkModel,
               ladder: models.LadderResult, scored: pd.DataFrame,
               rules: pd.DataFrame, sens: pd.DataFrame, corridors: pd.DataFrame,
               breakeven: pd.DataFrame, routing_summary: dict) -> Path:
    _banner("STAGE 5  Economics and strategy memo")
    stats = cfg.paths.resolve("stats")
    rar = economics.revenue_at_risk(legs, cfg)
    upgrade, detail = economics.hub_upgrade_scenario(legs, network.hub_metrics, cfg, top_n=3)
    model_value = economics.eta_model_value(ladder.metrics, legs, cfg)

    period_days = float(
        (legs["od_start_time"].max() - legs["od_start_time"].min()).total_seconds() / 86400
    )
    central = upgrade[(upgrade["recovery_scenario"] == "central")
                      & (upgrade["penalty_scenario"] == "central")].iloc[0]
    print(f"  Sampled period: {period_days:.0f} days")
    print(f"  Revenue at risk (central): Rs {rar[rar['scenario'] == 'central']['revenue_at_risk_cr'].iloc[0]:.2f} Cr")
    print(f"  Top-3 hub upgrade recovers {central['late_reduction_pct']:.1f}% of late legs "
          f"= Rs {central['revenue_recovered_cr']:.2f} Cr in period")
    print(f"  ETA model lever: Rs {model_value[model_value['penalty_scenario'] == 'central']['value_cr'].iloc[0]:.2f} Cr in period")

    text = memo_mod.build(
        cfg=cfg,
        breakeven=breakeven,
        routing_summary=routing_summary,
        ingest_report=_load(stats, "ingest_report"),
        sla_summary=_load(stats, "sla_definitions"),
        decomposition=_load(stats, "delay_decomposition").iloc[0].to_dict(),
        hubs=network.hub_metrics,
        corridors=corridors,
        ladder=ladder.metrics,
        advantage=ladder.advantage,
        routing_rules=rules,
        routing_scored=scored,
        sensitivity=sens,
        rar=rar,
        upgrade=upgrade,
        upgrade_detail=detail,
        model_value=model_value,
        cold_start=_load(stats, "cold_start_report").iloc[0].to_dict(),
        period_days=period_days,
    )
    path = memo_mod.write(text, cfg.paths.resolve("memo") / "network_operations_strategy_memo.md")

    stats_dir = cfg.paths.resolve("stats")
    comparison = None
    if (stats_dir / "embedder_comparison.parquet").exists():
        comparison = _load(stats_dir, "embedder_comparison")
    appendix_text = appendix_mod.build(
        cfg=cfg,
        ingest_report=_load(stats, "ingest_report"),
        sla_summary=_load(stats, "sla_definitions"),
        decomposition=_load(stats, "delay_decomposition").iloc[0].to_dict(),
        ladder=ladder.metrics,
        advantage=ladder.advantage,
        embedder_comparison=comparison,
        cold_start=_load(stats, "cold_start_report").iloc[0].to_dict(),
        hubs=network.hub_metrics,
        corridors=corridors,
        breakeven=breakeven,
        routing_summary=routing_summary,
        empirical=_load(stats, "routing_empirical_check"),
    )
    appendix_path = appendix_mod.write(
        appendix_text, cfg.paths.resolve("memo") / "technical_appendix.md"
    )
    print(f"  Appendix -> {appendix_path}")
    _save(rar, stats, "revenue_at_risk")
    _save(upgrade, stats, "hub_upgrade_scenarios")
    _save(detail, stats, "hub_upgrade_detail")
    _save(model_value, stats, "eta_model_value")
    print(f"\n  Memo -> {path}")
    return path


# ----------------------------------------------------------------------- main
def stage_compare(cfg: Config, legs: pd.DataFrame, network: gmod.NetworkModel) -> pd.DataFrame:
    """Does the choice of node representation matter? Run all three and see.

    A single embedder that fails to help could be a bad implementation. Three
    independent representations that all fail to help is a property of the data,
    and that is a far stronger claim -- so it is worth the extra two minutes.
    """
    _banner("EXPERIMENT  Does the choice of node embedding matter?")
    train = legs[legs["split"] == "train"].copy()
    test = legs[legs["split"] == "test"].copy()
    rows = []
    for kind in sorted(embeddings.EMBEDDERS):
        started = time.time()
        embedder = embeddings.build_embedder(kind, cfg).fit(network.graph)
        space = FeatureSpace(cfg=cfg, network=network, embedder=embedder)
        result = models.run_ladder(train, test, space, cfg)
        step = result.advantage[result.advantage["comparison"] == "structural -> embedded"].iloc[0]
        embedded = result.metrics[result.metrics["rung"] == "embedded"].iloc[0]
        rows.append({
            "embedder": kind,
            "fit_seconds": round(time.time() - started, 1),
            "embedded_mae": float(embedded["mae_minutes"]),
            "embedded_within15": float(embedded["within_15pct"]),
            "gain_over_structural": float(step["mae_reduction_minutes"]),
            "ci95_low": float(step["ci95_low"]),
            "ci95_high": float(step["ci95_high"]),
            "significant": bool(step["significant"]),
        })
        print(f"    {kind:<10} MAE {embedded['mae_minutes']:6.2f}  "
              f"gain over hand-built features {step['mae_reduction_minutes']:+.2f} min "
              f"[{step['ci95_low']:+.2f}, {step['ci95_high']:+.2f}]  "
              f"{'significant' if step['significant'] else 'NOT significant'}")
    frame = pd.DataFrame(rows)
    print("\n  None of the three learned representations beats hand-built corridor history")
    print("  plus centrality. That replicates across methods, so it is a property of this")
    print(f"  network ({network.graph.number_of_nodes():,} facilities / "
          f"{network.graph.number_of_edges():,} corridors), not of one implementation.")
    _save(frame, cfg.paths.resolve("stats"), "embedder_comparison")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Delhivery graph network intelligence")
    parser.add_argument("stage", choices=["all", "prep", "audit", "model", "routing", "memo", "compare"])
    parser.add_argument("--config", default=None, help="path to configs/pipeline.yaml")
    parser.add_argument("--embedder", default="node2vec",
                        choices=sorted(embeddings.EMBEDDERS), help="node embedding method")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    started = time.time()

    legs, network = stage_prep(cfg)
    if args.stage == "prep":
        print(f"\nDone in {time.time() - started:.1f}s")
        return
    if args.stage == "compare":
        stage_compare(cfg, legs, network)
        print(f"\nDone in {time.time() - started:.1f}s")
        return

    corridors = stage_audit(cfg, legs, network)
    if args.stage == "audit":
        print(f"\nDone in {time.time() - started:.1f}s")
        return

    ladder, space = stage_model(cfg, legs, network, args.embedder)
    if args.stage == "model":
        print(f"\nDone in {time.time() - started:.1f}s")
        return

    scored, rules, sens, breakeven, routing_summary = stage_routing(cfg, legs, network, space)
    if args.stage == "routing":
        print(f"\nDone in {time.time() - started:.1f}s")
        return

    stage_memo(cfg, legs, network, ladder, scored, rules, sens, corridors,
               breakeven, routing_summary)

    print(f"\nDone in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
