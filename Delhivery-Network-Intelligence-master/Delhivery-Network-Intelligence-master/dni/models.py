"""ETA models and the ablation ladder that makes "graph advantage" a measurement.

The brief is explicit that the graph advantage "must be measured, not claimed".
Measuring it properly requires two things that are easy to skip:

1. **A nested ladder with a shared feature base.** Each rung adds exactly one
   block of information on top of the previous one, using the identical model
   class, hyperparameters, target and split. A gap between two rungs is then
   attributable to the block that was added. The common failure is to hand the
   graph model corridor history through an edge attribute while denying it to
   the baseline, then report the difference as a win for the architecture.

2. **An uncertainty estimate on the gap.** A 2-minute MAE difference on a
   7,000-leg holdout may or may not be real. We paired-bootstrap the per-leg
   absolute errors and report a 95% interval on the difference, so the claim
   comes with a stated confidence rather than a decimal place.

The rungs:

  osrm        - the incumbent. OSRM's raw estimate, no model at all.
  trip        - tabular baseline: routing estimate + calendar. No graph.
  corridor    - + this corridor's own history (edge attributes)
  structural  - + where the endpoints sit in the network (centrality)
  embedded    - + learned node representations (node2vec / GraphSAGE / SVD)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMRegressor
    HAVE_LGBM = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAVE_LGBM = False

from .config import Config
from .features import FeatureSpace, assert_no_leakage

TARGET = "actual_minutes"
LADDER = ["trip", "corridor", "structural", "embedded"]
LADDER_LABELS = {
    "osrm": "OSRM raw estimate (incumbent, no model)",
    "trip": "Tabular baseline (routing estimate + calendar)",
    "corridor": "+ corridor history",
    "structural": "+ network position (centrality)",
    "embedded": "+ learned node embeddings",
}


def make_estimator(cfg: Config, numeric: list[str], categorical: list[str]) -> Pipeline:
    """One model class for every rung -- otherwise the ladder compares two things."""
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",
    )
    if HAVE_LGBM:
        model = LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0,
            objective="l1",            # optimise MAE, the metric we report
            random_state=cfg.seed, n_jobs=-1, verbose=-1,
        )
    else:  # pragma: no cover
        model = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, loss="absolute_error",
            random_state=cfg.seed,
        )
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict[str, float | str]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1.0, None)  # an ETA is positive
    ape = np.abs(y_pred - y_true) / np.clip(y_true, 1e-9, None)
    return {
        "model": label,
        "mae_minutes": float(mean_absolute_error(y_true, y_pred)),
        "rmse_minutes": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "within_15pct": float(np.mean(ape <= 0.15) * 100),
        "within_25pct": float(np.mean(ape <= 0.25) * 100),
        "median_ape_pct": float(np.median(ape) * 100),
        "bias_minutes": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }


def paired_bootstrap(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
    n_boot: int = 2000, seed: int = 42,
) -> dict[str, float]:
    """95% interval on (MAE_a - MAE_b), resampling legs in pairs.

    Pairing matters: both models saw the same legs, so the correlated part of
    the error cancels and the interval is far tighter -- and more honest -- than
    two independent intervals compared by eye.
    """
    rng = np.random.default_rng(seed)
    err_a = np.abs(np.asarray(pred_a, dtype=float) - y_true)
    err_b = np.abs(np.asarray(pred_b, dtype=float) - y_true)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = err_a[idx].mean() - err_b[idx].mean()
    observed = err_a.mean() - err_b.mean()
    return {
        "mae_delta": float(observed),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "prob_improvement": float(np.mean(diffs > 0)),
    }


@dataclass
class LadderResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    advantage: pd.DataFrame
    fitted: dict[str, Pipeline] = field(default_factory=dict)


def run_ladder(
    train_legs: pd.DataFrame,
    test_legs: pd.DataFrame,
    space: FeatureSpace,
    cfg: Config,
) -> LadderResult:
    train_matrix = space.build(train_legs)
    test_matrix = space.build(test_legs)
    y_train = train_matrix[TARGET].to_numpy(dtype=float)
    y_test = test_matrix[TARGET].to_numpy(dtype=float)

    rows = [evaluate(y_test, test_matrix["osrm_minutes"].to_numpy(), LADDER_LABELS["osrm"])]
    rows[0]["rung"] = "osrm"
    predictions = {
        "osrm": test_matrix["osrm_minutes"].to_numpy(dtype=float),
    }
    fitted: dict[str, Pipeline] = {}

    for rung in LADDER:
        numeric, categorical = space.columns(rung)
        assert_no_leakage(train_matrix, numeric + categorical)
        estimator = make_estimator(cfg, numeric, categorical)
        estimator.fit(train_matrix[numeric + categorical], y_train)
        pred = estimator.predict(test_matrix[numeric + categorical])
        row = evaluate(y_test, pred, LADDER_LABELS[rung])
        row["rung"] = rung
        row["n_features"] = len(numeric) + len(categorical)
        rows.append(row)
        predictions[rung] = np.clip(pred, 1.0, None)
        fitted[rung] = estimator

    metrics = pd.DataFrame(rows)
    metrics = metrics[["rung", "model", "n"] + [
        c for c in metrics.columns if c not in {"rung", "model", "n"}
    ]]

    # Incremental contribution of each block, each with its own interval.
    steps = [("osrm", "trip"), ("trip", "corridor"), ("corridor", "structural"),
             ("structural", "embedded"), ("trip", "embedded")]
    advantage_rows = []
    for base, better in steps:
        stats = paired_bootstrap(y_test, predictions[base], predictions[better], seed=cfg.seed)
        base_mae = float(np.mean(np.abs(predictions[base] - y_test)))
        advantage_rows.append({
            "comparison": f"{base} -> {better}",
            "question": _QUESTION[(base, better)],
            "mae_reduction_minutes": stats["mae_delta"],
            "mae_reduction_pct": 100 * stats["mae_delta"] / base_mae if base_mae else 0.0,
            "ci95_low": stats["ci_low"],
            "ci95_high": stats["ci_high"],
            "significant": bool(stats["ci_low"] > 0),
            "within15_gain_pp": (
                float(np.mean(np.abs(predictions[better] - y_test) / y_test <= 0.15) * 100)
                - float(np.mean(np.abs(predictions[base] - y_test) / y_test <= 0.15) * 100)
            ),
        })

    pred_frame = test_legs[[
        "trip_uuid", "source_center", "destination_center", "source_name",
        "destination_name", "route_type", "time_of_day", "osrm_minutes",
        "actual_minutes", "promised_minutes", "is_late",
    ]].copy()
    for rung, values in predictions.items():
        pred_frame[f"pred_{rung}"] = values

    return LadderResult(
        metrics=metrics,
        predictions=pred_frame,
        advantage=pd.DataFrame(advantage_rows),
        fitted=fitted,
    )


_QUESTION = {
    ("osrm", "trip"): "Does any model beat the routing engine we ship today?",
    ("trip", "corridor"): "Does knowing this lane's own history help?",
    ("corridor", "structural"): "Does network position add beyond lane history?",
    ("structural", "embedded"): "Do learned embeddings add beyond hand-built metrics?",
    ("trip", "embedded"): "Total graph advantage over a non-graph model.",
}
