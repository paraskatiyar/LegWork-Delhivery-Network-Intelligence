"""Raw scan records -> one clean row per delivery leg.

The dataset is at *scan* grain: 144,867 rows describing 26,369 origin-
destination legs. Getting from one to the other is the single highest-leverage
decision in this project, and it is where naive pipelines go wrong:

  * Modelling the raw rows treats `actual_time` (which is cumulative for the
    leg and therefore repeated on every scan of that leg) as if it were a
    per-row observation. Error metrics then get computed over duplicated
    targets and are not comparable to leg-level metrics.
  * Filtering to `is_cutoff == False` finds the leg-summary rows, which is
    correct -- but silently loses 251 legs that have no summary row, and
    inherits one duplicated key.
  * Summing `segment_actual_time` reconstructs the leg independently, and
    disagrees with the summary row on ~65% of legs (median 1 minute).

We do both, reconcile them, and record the disagreement as a data-quality
artifact instead of quietly picking one.

UNITS: every duration in this project is MINUTES and every distance is KM.
`assert_units` is called after extraction so a unit slip fails the run rather
than surfacing as an implausible number in the memo.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

LEG_KEYS = ["trip_uuid", "source_center", "destination_center"]

EXPECTED_COLUMNS = {
    "data", "trip_creation_time", "route_schedule_uuid", "route_type", "trip_uuid",
    "source_center", "source_name", "destination_center", "destination_name",
    "od_start_time", "od_end_time", "start_scan_to_end_scan", "is_cutoff",
    "cutoff_factor", "cutoff_timestamp", "actual_distance_to_destination",
    "actual_time", "osrm_time", "osrm_distance", "factor", "segment_actual_time",
    "segment_osrm_time", "segment_osrm_distance", "segment_factor",
}

#: Columns that encode the outcome and are therefore unavailable when an ETA
#: must be quoted (at dispatch). Any of these reaching a model matrix is a bug,
#: not a modelling choice -- see features.assert_no_leakage.
FORBIDDEN_FEATURES = {
    "actual_time", "actual_minutes", "actual_minutes_segsum", "actual_minutes_summary",
    "actual_minutes_cummax", "factor", "delay_ratio", "delay_ratio_w", "delay_minutes",
    "segment_actual_time", "segment_factor", "od_end_time", "is_late", "is_late_raw",
    "actual_distance_to_destination",   # remaining distance, known only in transit
    "start_scan_to_end_scan",           # wall-clock leg duration; known at completion
    "cutoff_factor", "is_cutoff", "cutoff_timestamp",  # post-hoc scan bookkeeping
    "reconstruction_gap_min", "scan_count",  # scan_count is unknown before the leg runs
}


@dataclass
class IngestReport:
    raw_rows: int
    legs_total: int
    legs_with_summary_row: int
    legs_without_summary_row: int
    duplicate_summary_keys: int
    reconciliation_exact_pct: float
    reconciliation_median_abs_min: float
    reconciliation_p95_abs_min: float
    legs_after_filters: int
    dropped_by_filters: int
    train_legs: int
    test_legs: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.__dict__])


def _time_of_day(hour: pd.Series, cfg: Config) -> pd.Series:
    return pd.cut(
        hour, bins=list(cfg.graph.time_bins), labels=list(cfg.graph.time_labels), ordered=True
    ).astype("string")


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place delivery_data.csv under data/raw/."
        )
    df = pd.read_csv(path, low_memory=False)
    missing = sorted(EXPECTED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    for col in ("trip_creation_time", "od_start_time", "od_end_time", "cutoff_timestamp"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["is_cutoff"] = (
        df["is_cutoff"].astype("string").str.lower()
        .map({"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False})
        .fillna(False)
    )
    return df


def build_legs(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, IngestReport]:
    """Collapse scan rows to one row per leg, reconciling both reconstructions."""
    raw_rows = len(df)

    # -- Reconstruction A: sum the per-segment deltas ------------------------
    grouped = df.groupby(LEG_KEYS, sort=False, observed=True)
    legs = grouped.agg(
        data=("data", "first"),
        route_type=("route_type", "first"),
        route_schedule_uuid=("route_schedule_uuid", "first"),
        source_name=("source_name", "first"),
        destination_name=("destination_name", "first"),
        trip_creation_time=("trip_creation_time", "min"),
        od_start_time=("od_start_time", "min"),
        od_end_time=("od_end_time", "max"),
        scan_count=("segment_actual_time", "size"),
        actual_minutes_segsum=("segment_actual_time", "sum"),
        osrm_minutes_segsum=("segment_osrm_time", "sum"),
        osrm_km_segsum=("segment_osrm_distance", "sum"),
        actual_minutes_cummax=("actual_time", "max"),
    ).reset_index()

    # -- Reconstruction B: the leg-summary scan (is_cutoff == False) ---------
    summary = df.loc[~df["is_cutoff"], LEG_KEYS + ["actual_time", "osrm_time", "osrm_distance"]]
    duplicate_summary_keys = int(summary.duplicated(subset=LEG_KEYS).sum())
    summary = summary.drop_duplicates(subset=LEG_KEYS, keep="last").rename(
        columns={
            "actual_time": "actual_minutes_summary",
            "osrm_time": "osrm_minutes_summary",
            "osrm_distance": "osrm_km_summary",
        }
    )
    legs = legs.merge(summary, on=LEG_KEYS, how="left")

    has_summary = legs["actual_minutes_summary"].notna()
    delta = (legs.loc[has_summary, "actual_minutes_summary"]
             - legs.loc[has_summary, "actual_minutes_segsum"]).abs()

    # -- Reconcile: prefer the summary row, fall back to the segment sum -----
    legs["actual_minutes"] = legs["actual_minutes_summary"].fillna(legs["actual_minutes_segsum"])
    legs["osrm_minutes"] = legs["osrm_minutes_summary"].fillna(legs["osrm_minutes_segsum"])
    legs["osrm_km"] = legs["osrm_km_summary"].fillna(legs["osrm_km_segsum"])
    legs["reconstruction_source"] = np.where(has_summary, "summary_row", "segment_sum")
    legs["reconstruction_gap_min"] = (
        legs["actual_minutes_summary"] - legs["actual_minutes_segsum"]
    ).abs()

    # -- Filters -------------------------------------------------------------
    before = len(legs)
    legs = legs[
        (legs["actual_minutes"] > 0)
        & (legs["osrm_minutes"] > 0)
        & (legs["osrm_km"] >= 0)
        & (legs["source_center"] != legs["destination_center"])
    ].copy()
    dropped = before - len(legs)

    # -- Derived fields (targets and descriptors, NOT features) --------------
    reference = legs["od_start_time"].fillna(legs["trip_creation_time"])
    legs["start_hour"] = reference.dt.hour
    legs["day_of_week"] = reference.dt.day_name()
    legs["is_weekend"] = reference.dt.dayofweek.isin([5, 6]).astype("int8")
    legs["time_of_day"] = _time_of_day(legs["start_hour"], cfg)
    legs["delay_ratio"] = legs["actual_minutes"] / legs["osrm_minutes"].clip(lower=1e-6)
    legs["delay_minutes"] = legs["actual_minutes"] - legs["osrm_minutes"]
    lo, hi = legs["delay_ratio"].quantile([cfg.graph.winsor_lower, cfg.graph.winsor_upper])
    legs["delay_ratio_w"] = legs["delay_ratio"].clip(lo, hi)
    legs["split"] = np.where(
        legs["data"].astype("string").str.lower().eq("training"), "train", "test"
    )
    legs["source_name"] = legs["source_name"].fillna(legs["source_center"])
    legs["destination_name"] = legs["destination_name"].fillna(legs["destination_center"])
    legs["corridor_id"] = legs["source_center"] + "->" + legs["destination_center"]

    report = IngestReport(
        raw_rows=raw_rows,
        legs_total=before,
        legs_with_summary_row=int(has_summary.sum()),
        legs_without_summary_row=int((~has_summary).sum()),
        duplicate_summary_keys=duplicate_summary_keys,
        reconciliation_exact_pct=float((delta < 1e-6).mean() * 100) if len(delta) else 0.0,
        reconciliation_median_abs_min=float(delta.median()) if len(delta) else 0.0,
        reconciliation_p95_abs_min=float(delta.quantile(0.95)) if len(delta) else 0.0,
        legs_after_filters=len(legs),
        dropped_by_filters=dropped,
        train_legs=int((legs["split"] == "train").sum()),
        test_legs=int((legs["split"] == "test").sum()),
    )
    return legs.reset_index(drop=True), report


def assert_units(legs: pd.DataFrame) -> None:
    """Fail loudly if durations stop looking like minutes.

    One reviewed submission reported an ETA MAE of "12.55 hours" on legs whose
    median duration is 84 minutes; another divided minutes by 3600 and
    concluded FTL saves ~4 seconds per leg, then wrote that up as a finding.
    Both are cheap to catch mechanically, so we catch them.
    """
    median = float(legs["actual_minutes"].median())
    if not 10 <= median <= 600:
        raise AssertionError(
            f"actual_minutes median is {median:.2f}; expected 10-600 minutes. "
            "A unit conversion is wrong somewhere upstream."
        )
    if (legs["actual_minutes"] <= 0).any():
        raise AssertionError("Non-positive durations survived filtering.")
    ratio = float(legs["delay_ratio"].median())
    if not 0.5 <= ratio <= 10:
        raise AssertionError(f"Implausible median delay ratio {ratio:.2f}.")
