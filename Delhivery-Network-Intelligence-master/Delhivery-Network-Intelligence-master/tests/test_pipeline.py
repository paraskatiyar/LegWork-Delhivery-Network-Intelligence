"""Correctness guards.

These are not coverage theatre. Each test encodes a specific failure that was
observed in real submissions of this problem, so that the same mistake cannot
be made here without the suite going red:

  * a model matrix containing a column derived from the target
  * a graph fitted on data that includes the holdout
  * durations silently changing units between modules
  * a hub-upgrade scenario whose arithmetic can only ever return zero
  * an SLA rule that flags almost every shipment and is then used to rank

Run with:  python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dni import economics, graph as gmod, ingest, models, routing, sla
from dni.config import Config
from dni.features import assert_no_leakage
from dni.ingest import FORBIDDEN_FEATURES


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config()


def _synthetic_legs(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A small network with a known structure, so expectations are checkable."""
    rng = np.random.default_rng(seed)
    facilities = [f"F{i:02d}" for i in range(12)]
    src = rng.choice(facilities, n)
    dst = rng.choice(facilities, n)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    n = len(src)
    osrm = rng.uniform(20, 300, n)
    ratio = rng.uniform(1.1, 2.6, n)
    start = pd.Timestamp("2018-09-12") + pd.to_timedelta(rng.integers(0, 480, n), unit="h")
    frame = pd.DataFrame({
        "trip_uuid": [f"t{i}" for i in range(n)],
        "source_center": src,
        "destination_center": dst,
        "source_name": [f"{s} Depot (State)" for s in src],
        "destination_name": [f"{d} Depot (State)" for d in dst],
        "route_type": rng.choice(["FTL", "Carting"], n),
        "osrm_minutes": osrm,
        "osrm_km": osrm * rng.uniform(0.6, 1.4, n),
        "actual_minutes": osrm * ratio,
        "od_start_time": start,
        "trip_creation_time": start,
        "start_hour": start.hour,
        "day_of_week": start.day_name(),
        "is_weekend": start.dayofweek.isin([5, 6]).astype("int8"),
        "time_of_day": pd.Series(rng.choice(["morning", "evening", "night"], n)).astype("string"),
        "split": np.where(np.arange(n) < int(n * 0.7), "train", "test"),
    })
    frame["corridor_id"] = frame["source_center"] + "->" + frame["destination_center"]
    frame["delay_ratio"] = frame["actual_minutes"] / frame["osrm_minutes"]
    frame["delay_minutes"] = frame["actual_minutes"] - frame["osrm_minutes"]
    frame["delay_ratio_w"] = frame["delay_ratio"]
    return frame


# ---------------------------------------------------------------- leakage
def test_forbidden_features_are_rejected():
    """The firewall must reject an outcome-derived column, not warn about it."""
    matrix = pd.DataFrame({"osrm_minutes": [1.0], "actual_minutes": [2.0]})
    with pytest.raises(AssertionError, match="Leakage firewall"):
        assert_no_leakage(matrix, ["osrm_minutes", "actual_minutes"])


def test_target_reconstructible_columns_are_forbidden():
    """delay = actual - osrm; if both terms are features the target is arithmetic.

    This is the exact failure that produced an "84% improvement" in one
    reviewed submission, carried straight into a production recommendation.
    """
    for column in ("actual_minutes", "delay_minutes", "delay_ratio", "factor"):
        assert column in FORBIDDEN_FEATURES, f"{column} must never be a feature"


def test_post_hoc_columns_are_forbidden():
    """Fields known only after the shipment completes cannot price an ETA."""
    for column in ("start_scan_to_end_scan", "cutoff_factor", "is_cutoff",
                   "actual_distance_to_destination"):
        assert column in FORBIDDEN_FEATURES


def test_clean_feature_list_passes():
    matrix = pd.DataFrame({"osrm_minutes": [1.0], "osrm_km": [2.0], "route_type": ["FTL"]})
    assert_no_leakage(matrix, ["osrm_minutes", "osrm_km", "route_type"])


# ------------------------------------------------------------ train-only fit
def test_network_refuses_test_legs(cfg):
    """Fitting the graph on anything but training legs must be impossible."""
    legs = _synthetic_legs()
    calibrated = sla.SLACalibration.fit(legs[legs["split"] == "train"], cfg).apply(legs, cfg)
    with pytest.raises(ValueError, match="training split only"):
        gmod.NetworkModel(cfg).fit(calibrated)


def test_network_accepts_train_legs(cfg):
    legs = _synthetic_legs()
    calibrated = sla.SLACalibration.fit(legs[legs["split"] == "train"], cfg).apply(legs, cfg)
    train = calibrated[calibrated["split"] == "train"]
    network = gmod.NetworkModel(cfg).fit(train)
    assert network.graph.number_of_nodes() > 0
    # No facility that appears only in the holdout may be in the graph.
    test_only = set(calibrated[calibrated["split"] == "test"]["source_center"]) - set(
        train["source_center"]) - set(train["destination_center"])
    assert not (test_only & set(network.graph.nodes()))


def test_cold_start_is_flagged_not_faked(cfg):
    """Unseen corridors must be marked, not silently given a plausible number."""
    legs = _synthetic_legs()
    calibrated = sla.SLACalibration.fit(legs[legs["split"] == "train"], cfg).apply(legs, cfg)
    train = calibrated[calibrated["split"] == "train"]
    network = gmod.NetworkModel(cfg).fit(train)
    unseen = pd.DataFrame({
        "source_center": ["ZZZ"], "destination_center": ["YYY"],
        "route_type": ["FTL"], "time_of_day": ["morning"],
    })
    edges = network.edge_features(unseen)
    assert edges["edge_is_cold_start"].iloc[0] == 1
    assert pd.isna(edges["edge_median_delay_ratio"].iloc[0])


# ------------------------------------------------------------------- units
def test_unit_assertion_catches_seconds():
    """Minutes mistaken for seconds must fail the run, not reach a memo."""
    legs = _synthetic_legs()
    legs["actual_minutes"] = legs["actual_minutes"] * 60  # pretend seconds
    with pytest.raises(AssertionError, match="expected 10-600 minutes"):
        ingest.assert_units(legs)


def test_unit_assertion_catches_hours():
    legs = _synthetic_legs()
    legs["actual_minutes"] = legs["actual_minutes"] / 60  # pretend hours
    with pytest.raises(AssertionError, match="expected 10-600 minutes"):
        ingest.assert_units(legs)


def test_unit_assertion_passes_on_minutes():
    ingest.assert_units(_synthetic_legs())


# --------------------------------------------------------------------- SLA
def test_calibrated_rule_is_more_selective_than_raw(cfg):
    """If the calibrated rule also flagged ~everything it would be no better."""
    legs = _synthetic_legs()
    calibrated = sla.SLACalibration.fit(legs[legs["split"] == "train"], cfg).apply(legs, cfg)
    assert calibrated["is_late"].mean() < calibrated["is_late_raw"].mean()
    assert calibrated["is_late"].mean() < 0.6, "a ranking metric must not fire on most legs"


def test_delay_decomposition_sums_to_whole(cfg):
    legs = _synthetic_legs()
    calibrated = sla.SLACalibration.fit(legs[legs["split"] == "train"], cfg).apply(legs, cfg)
    parts = sla.decompose_delay(calibrated)
    total = (parts["systematic_share_pct"] + parts["tolerance_share_pct"]
             + parts["operational_share_pct"])
    assert total == pytest.approx(100.0, abs=0.01), "the three parts must partition the gap"
    for key in ("systematic_share_pct", "tolerance_share_pct", "operational_share_pct"):
        assert parts[key] >= 0
    assert parts["systematic_forecast_bias_minutes"] <= parts["total_gap_vs_osrm_minutes"]


# --------------------------------------------------------------- economics
def test_hub_upgrade_recovers_something(cfg):
    """A recovery model that can only ever return zero is a broken model.

    Scaling every leg's excess by a constant factor has exactly that property:
    no leg crosses back inside its promise unless the factor is 1.0.
    """
    legs = _synthetic_legs(600)
    calibrated = sla.SLACalibration.fit(legs[legs["split"] == "train"], cfg).apply(legs, cfg)
    network = gmod.NetworkModel(cfg).fit(calibrated[calibrated["split"] == "train"])
    scenarios, detail = economics.hub_upgrade_scenario(calibrated, network.hub_metrics, cfg)
    assert len(scenarios) > 0
    assert scenarios["late_legs_recovered"].max() > 0, "recovery model returns zero for all scenarios"
    assert (scenarios["late_reduction_pct"] <= 100).all()
    assert len(detail) >= 3


def test_revenue_scales_monotonically_with_penalty(cfg):
    legs = _synthetic_legs()
    calibrated = sla.SLACalibration.fit(legs[legs["split"] == "train"], cfg).apply(legs, cfg)
    rar = economics.revenue_at_risk(calibrated, cfg).sort_values("penalty_share_of_revenue")
    assert rar["revenue_at_risk_inr"].is_monotonic_increasing


# ------------------------------------------------------------------ routing
def test_cost_model_has_a_crossover(cfg):
    """FTL must win somewhere, or the framework is just 'always Carting'."""
    km = pd.Series([10.0, 100.0, 500.0])
    mode_ftl = pd.Series(["FTL"] * 3)
    mode_cart = pd.Series(["Carting"] * 3)
    ftl = routing.transport_cost(mode_ftl, km, cfg)
    cart = routing.transport_cost(mode_cart, km, cfg)
    assert ftl.iloc[0] > cart.iloc[0], "FTL should be dearer on a short hop"
    assert ftl.iloc[2] < cart.iloc[2], "FTL should win on a long haul"


def test_metrics_are_sane():
    y = np.array([100.0, 200.0, 300.0])
    perfect = models.evaluate(y, y.copy(), "perfect")
    assert perfect["mae_minutes"] == pytest.approx(0.0)
    assert perfect["within_15pct"] == pytest.approx(100.0)


def test_paired_bootstrap_detects_no_difference():
    """Identical predictions must not produce a 'significant' improvement."""
    rng = np.random.default_rng(0)
    y = rng.uniform(50, 500, 500)
    pred = y + rng.normal(0, 20, 500)
    stats = models.paired_bootstrap(y, pred, pred.copy(), n_boot=300)
    assert stats["mae_delta"] == pytest.approx(0.0, abs=1e-9)
    assert stats["ci_low"] <= 0 <= stats["ci_high"]


def test_paired_bootstrap_detects_real_difference():
    rng = np.random.default_rng(0)
    y = rng.uniform(50, 500, 500)
    worse = y + rng.normal(0, 60, 500)
    better = y + rng.normal(0, 10, 500)
    stats = models.paired_bootstrap(y, worse, better, n_boot=300)
    assert stats["ci_low"] > 0, "a genuine improvement should clear zero"
