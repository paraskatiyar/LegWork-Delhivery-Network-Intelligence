"""Single source of truth for pipeline configuration.

Every tunable lives here or in ``configs/pipeline.yaml``. Nothing in the
pipeline reads a hard-coded path or threshold, so a reviewer can change an
assumption in one place and re-run to see the whole downstream effect --
including the strategy memo, which is generated from these artifacts rather
than written by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:  # PyYAML is optional; the defaults below are authoritative without it.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parent.parent
    raw_csv: Path = Path("data/raw/delivery_data.csv")
    interim: Path = Path("outputs/interim")
    processed: Path = Path("outputs/processed")
    graphs: Path = Path("outputs/graphs")
    models: Path = Path("outputs/models")
    stats: Path = Path("outputs/stats")
    figures: Path = Path("outputs/figures")
    memo: Path = Path("outputs/memo")

    def resolve(self, attr: str) -> Path:
        value = getattr(self, attr)
        return value if value.is_absolute() else self.root / value


@dataclass(frozen=True)
class SLAPolicy:
    """How we decide a leg is late.

    ``ratio_threshold`` of 1.20 applied to raw OSRM is the threshold named in
    the brief, and we report it -- but on this network it flags ~94% of legs,
    which makes it useless for prioritisation. ``calibrated`` re-bases the
    promise on what the network actually achieves (a corridor-level quantile
    of the realised delay ratio), so "late" means late against a promise the
    operation could plausibly keep. Both are computed; the memo leads with the
    calibrated one and explains the difference.
    """

    ratio_threshold: float = 1.20
    calibrate: bool = True
    calibration_quantile: float = 0.50
    min_legs_for_corridor_promise: int = 5
    #: A corridor must carry at least this many legs to enter a priority list.
    min_legs_for_priority: int = 30
    #: Median ratios above this are treated as data artifacts (mis-scans,
    #: outages, OSRM mis-calibration on unusual geometry), not as corridors to
    #: fund. They are reported separately for data-quality follow-up.
    artifact_ratio_ceiling: float = 6.0


@dataclass(frozen=True)
class GraphPolicy:
    min_legs_per_edge: int = 3
    time_bins: tuple[int, ...] = (-1, 5, 11, 16, 20, 23)
    time_labels: tuple[str, ...] = ("late_night", "morning", "afternoon", "evening", "night")
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99


@dataclass(frozen=True)
class EmbeddingPolicy:
    dim: int = 24
    # node2vec
    walk_length: int = 20
    walks_per_node: int = 10
    p: float = 1.0
    q: float = 0.5          # q<1 biases toward outward exploration (structural roles)
    window: int = 5
    negative: int = 5
    epochs: int = 3
    lr: float = 0.02
    # GraphSAGE
    sage_hidden: int = 32
    sage_epochs: int = 60
    sage_lr: float = 0.01
    seed: int = 42


@dataclass(frozen=True)
class CostModel:
    """FTL vs Carting economics.

    These are *assumptions*, not measurements -- the dataset carries no cost
    column. They are stated here, surfaced in every output, and swept in a
    sensitivity analysis so a reader can see exactly which recommendations
    survive a different rate card. No external benchmark is cited, because we
    do not have one.
    """

    ftl_fixed_inr: float = 2500.0
    ftl_per_km_inr: float = 32.0
    carting_fixed_inr: float = 450.0
    carting_per_km_inr: float = 42.0
    carting_handling_inr: float = 250.0
    value_of_time_inr_per_min: float = 15.0
    late_penalty_inr_per_min: float = 35.0
    sensitivity_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)


@dataclass(frozen=True)
class RevenueModel:
    """Revenue-at-risk assumptions, all explicit and all swept."""

    revenue_per_leg_inr: float = 450.0
    penalty_share_of_revenue: float = 0.10
    penalty_share_low: float = 0.06
    penalty_share_high: float = 0.14
    excess_delay_recovery_rate: float = 0.50
    recovery_low: float = 0.30
    recovery_high: float = 0.65


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    sla: SLAPolicy = field(default_factory=SLAPolicy)
    graph: GraphPolicy = field(default_factory=GraphPolicy)
    embed: EmbeddingPolicy = field(default_factory=EmbeddingPolicy)
    cost: CostModel = field(default_factory=CostModel)
    revenue: RevenueModel = field(default_factory=RevenueModel)
    seed: int = 42

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        if path is None or yaml is None:
            return cls()
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        section_types = {
            "paths": Paths, "sla": SLAPolicy, "graph": GraphPolicy,
            "embed": EmbeddingPolicy, "cost": CostModel, "revenue": RevenueModel,
        }
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in section_types and isinstance(value, dict):
                if key == "paths":
                    value = {k: Path(v) for k, v in value.items()}
                kwargs[key] = section_types[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
