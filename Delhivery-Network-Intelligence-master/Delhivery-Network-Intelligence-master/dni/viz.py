"""Figures. Each one exists to answer a question a reader will actually ask.

House rules, applied throughout:
  * A colour encodes one variable, and the legend says which.
  * Facilities are labelled by name, never by centre code. "IND000000ACB" is
    unreadable to the operations leader who has to act on it; "Gurgaon Bilaspur"
    is a place they can call.
  * When a threshold flags 94% of the data, drawing it in red communicates
    nothing. The network plot therefore colours by *contribution to excess*,
    not by the binary chronic flag.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

INK = "#111827"
MUTED = "#9ca3af"
ACCENT = "#dc2626"
COOL = "#2563eb"
WARM = "#f59e0b"


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", color=INK, loc="left", pad=12)
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel(ylabel, fontsize=10, color=INK)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    ax.tick_params(colors=INK, labelsize=9)


def _short(name: str, width: int = 30) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


def sla_definition_chart(summary: pd.DataFrame, decomposition: dict, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rows = summary.iloc[:2]
    labels = ["Brief's rule\n(actual > 1.2x OSRM)", "Calibrated\n(vs corridor's own promise)"]
    axes[0].bar(labels, rows["late_rate_pct"], color=[MUTED, ACCENT], width=0.55)
    for i, v in enumerate(rows["late_rate_pct"]):
        axes[0].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold", color=INK)
    axes[0].set_ylim(0, 105)
    _style(axes[0], "The 1.2x rule flags almost everything", ylabel="Share of legs late (%)")
    axes[0].grid(axis="y", alpha=0.25, linestyle=":"); axes[0].grid(axis="x", visible=False)

    parts = [decomposition["systematic_share_pct"], decomposition["operational_share_pct"]]
    axes[1].barh(["Systematic\nforecast bias", "Operational\nexcess"], parts,
                 color=[COOL, ACCENT], height=0.5)
    for i, v in enumerate(parts):
        axes[1].text(v + 1, i, f"{v:.1f}%", va="center", fontweight="bold", color=INK)
    axes[1].set_xlim(0, 100)
    _style(axes[1], "Where the gap against OSRM actually comes from",
           xlabel="Share of total delay minutes vs OSRM (%)")
    fig.suptitle(
        "Most of the delay is a forecasting problem, not a concrete problem",
        fontsize=15, fontweight="bold", color=INK, x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def network_map(graph: nx.DiGraph, hubs: pd.DataFrame, corridors: pd.DataFrame,
                path: Path, top_hubs: int = 6, top_corridors: int = 25) -> Path:
    """The network, drawn so the eye lands on the hubs that matter.

    A spring layout over 1,300 facilities collapses into a hairball with every
    important label stacked in the centre. Three fixes: spread the layout, draw
    the priority corridors *on top* of the nodes so they are not buried, and
    push each hub label radially outward from the centroid with a leader line,
    so labels sit in the sparse outer ring instead of over each other.
    """
    fig, ax = plt.subplots(figsize=(17, 13))
    pos = nx.spring_layout(
        graph, seed=42, weight="leg_count",
        k=2.6 / np.sqrt(graph.number_of_nodes()), iterations=150,
    )

    priority = corridors[corridors["is_credible_priority"] == 1].head(top_corridors)
    hot = {
        (s, d) for s, d in zip(priority["source_center"], priority["destination_center"])
        if graph.has_edge(s, d)
    }
    top = hubs.head(top_hubs)

    scores = hubs.set_index("facility")["chokepoint_score"].to_dict()
    sizes = [20 + 2600 * scores.get(n, 0.0) for n in graph.nodes()]
    colors = [scores.get(n, 0.0) for n in graph.nodes()]

    normal = [(u, v) for u, v in graph.edges() if (u, v) not in hot]
    nx.draw_networkx_edges(graph, pos, edgelist=normal, ax=ax, edge_color="#d7dde5",
                           width=0.35, alpha=0.45, arrows=False)
    nodes = nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=sizes, node_color=colors,
                                   cmap="YlOrRd", alpha=0.92, linewidths=0.3,
                                   edgecolors="white")
    nodes.set_zorder(2)
    # Priority corridors go last so they read above the node cloud.
    if hot:
        drawn = nx.draw_networkx_edges(
            graph, pos, edgelist=sorted(hot), ax=ax, edge_color=ACCENT, width=2.4,
            alpha=0.95, arrows=True, arrowsize=11, connectionstyle="arc3,rad=0.10",
            node_size=0,
        )
        for artist in (drawn if isinstance(drawn, list) else [drawn]):
            artist.set_zorder(3)

    # Label placement. The chokepoint hubs all sit near the centre of the
    # layout, so their radial directions bunch together and the labels collide.
    # Instead: sort them by their true angle from the centroid, then deal them
    # evenly spaced slots on a ring in that same order. Order is preserved, so
    # each leader line still points the way the eye expects, and no two labels
    # can overlap.
    coords = np.array(list(pos.values()))
    centre = coords.mean(axis=0)
    span = np.abs(coords - centre).max()

    placed = [(row, np.array(pos[row["facility"]]))
              for _, row in top.iterrows() if row["facility"] in pos]
    placed.sort(key=lambda item: np.arctan2(*(item[1] - centre)[::-1]))
    count = max(len(placed), 1)
    start = np.arctan2(*(placed[0][1] - centre)[::-1]) if placed else 0.0

    for slot, (row, point) in enumerate(placed):
        angle = start + 2 * np.pi * slot / count
        anchor = centre + np.array([np.cos(angle), np.sin(angle)]) * span * 1.12
        ax.annotate(
            _short(str(row["facility_name"]).split("(")[0].strip().replace("_", " "), 22),
            xy=point, xytext=anchor, textcoords="data",
            fontsize=10, fontweight="bold", color=INK, zorder=5,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=ACCENT, lw=1.1, alpha=0.96),
            arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.9, alpha=0.75,
                            shrinkA=2, shrinkB=4),
        )
    cbar = fig.colorbar(nodes, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Chokepoint score (centrality x volume x excess delay)", fontsize=9)
    ax.set_title(
        "Delhivery corridor network: where service failure concentrates\n"
        f"Node size and colour = chokepoint score  |  Red = top {top_corridors} "
        "corridors by excess-delay contribution (volume-credible only)",
        fontsize=14, fontweight="bold", color=INK, loc="left", pad=14,
    )
    ax.margins(0.12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def hub_ranking(hubs: pd.DataFrame, path: Path, top_n: int = 10) -> Path:
    top = hubs.head(top_n).iloc[::-1]
    labels = [_short(str(n).split("(")[0].strip(), 26) for n in top["facility_name"]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    axes[0].barh(labels, top["chokepoint_score"], color=ACCENT, alpha=0.85)
    _style(axes[0], "Structural criticality", xlabel="Chokepoint score")
    axes[1].barh(labels, top["sla_contribution_pct"], color=COOL, alpha=0.85)
    _style(axes[1], "Share of network excess delay", xlabel="SLA-breach contribution (%)")
    fig.suptitle("The five hubs worth fixing first", fontsize=15, fontweight="bold",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def corridor_ranking(corridors: pd.DataFrame, path: Path, top_n: int = 12) -> Path:
    top = corridors[corridors["is_credible_priority"] == 1].head(top_n).iloc[::-1]
    labels = [_short(c.replace(" -> ", " > "), 46) for c in top["corridor"]]
    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.barh(labels, top["sla_contribution_pct"], color=ACCENT, alpha=0.85)
    for bar, ratio, legs in zip(bars, top["median_delay_ratio"], top["legs"]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{ratio:.2f}x  ({int(legs):,} legs)", va="center", fontsize=8, color=INK)
    _style(ax, "Corridors contributing most excess delay",
           xlabel="Share of network excess delay minutes (%)")
    ax.set_title(
        "Corridors contributing most excess delay\n"
        "Ranked by contribution, not by ratio: low-volume lanes with extreme "
        "ratios are data artifacts, not investment cases",
        fontsize=13, fontweight="bold", color=INK, loc="left", pad=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def ladder_chart(metrics: pd.DataFrame, advantage: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    labels = [_short(m, 34) for m in metrics["model"]]
    colours = [MUTED] + [COOL] * (len(metrics) - 2) + [ACCENT]

    axes[0].barh(labels[::-1], metrics["mae_minutes"][::-1], color=colours[::-1], alpha=0.9)
    for i, v in enumerate(metrics["mae_minutes"][::-1]):
        axes[0].text(v + 1, i, f"{v:.1f}", va="center", fontsize=9, fontweight="bold", color=INK)
    _style(axes[0], "Prediction error", xlabel="MAE (minutes) - lower is better")

    axes[1].barh(labels[::-1], metrics["within_15pct"][::-1], color=colours[::-1], alpha=0.9)
    for i, v in enumerate(metrics["within_15pct"][::-1]):
        axes[1].text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold", color=INK)
    _style(axes[1], "Business metric", xlabel="Legs predicted within 15% of actual (%)")

    fig.suptitle(
        "Each rung adds one block of information to the same model, features and split\n"
        "so the gap between rungs is attributable to that block and nothing else",
        fontsize=13, fontweight="bold", color=INK, x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def advantage_chart(advantage: pd.DataFrame, path: Path) -> Path:
    # Drop the osrm -> trip step. It is worth ~66 minutes, which on a shared
    # axis flattens the three graph blocks -- the actual subject of this chart --
    # into invisible slivers. That comparison has its own panel in the ladder
    # figure; here the question is what the *graph* adds.
    skip = {"trip -> embedded", "osrm -> trip"}
    df = advantage[~advantage["comparison"].isin(skip)].iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    y = np.arange(len(df))
    ax.barh(y, df["mae_reduction_minutes"],
            color=[ACCENT if s else MUTED for s in df["significant"]], alpha=0.9, height=0.55)
    ax.errorbar(
        df["mae_reduction_minutes"], y,
        xerr=[df["mae_reduction_minutes"] - df["ci95_low"],
              df["ci95_high"] - df["mae_reduction_minutes"]],
        fmt="none", ecolor=INK, elinewidth=1.4, capsize=5,
    )
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([_short(q, 52) for q in df["question"]], fontsize=9)
    _style(ax, "", xlabel="MAE reduction (minutes), with 95% paired-bootstrap interval")
    ax.set_title(
        "What each block of graph information is actually worth\n"
        "Red clears zero. Grey does not, and is therefore not evidence of an improvement.",
        fontsize=13, fontweight="bold", color=INK, loc="left", pad=12,
    )
    for value, y_pos, sig in zip(df["mae_reduction_minutes"], y, df["significant"]):
        ax.text(value + 0.15, y_pos, f"{value:+.2f} min", va="center", ha="left",
                fontsize=9, fontweight="bold", color=INK if sig else MUTED)
    ax.margins(x=0.20)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def routing_chart(scored: pd.DataFrame, sens: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    band = scored[scored["in_overlap_band"]]
    mix = (
        band.groupby("distance_band", observed=True)
        .agg(today=("observed_route_type", lambda s: float((s == "FTL").mean() * 100)),
             recommended=("recommended_route_type", lambda s: float((s == "FTL").mean() * 100)))
        .reset_index()
    )
    x = np.arange(len(mix))
    axes[0].bar(x - 0.2, mix["today"], width=0.4, label="FTL share today", color=MUTED)
    axes[0].bar(x + 0.2, mix["recommended"], width=0.4, label="FTL share recommended", color=ACCENT)
    axes[0].set_xticks(x); axes[0].set_xticklabels(mix["distance_band"], fontsize=9)
    axes[0].legend(frameon=False, fontsize=9)
    _style(axes[0], "Where the framework would change dispatch", ylabel="FTL share (%)")
    axes[0].grid(axis="y", alpha=0.25, linestyle=":"); axes[0].grid(axis="x", visible=False)

    for parameter, colour in zip(sens["parameter"].unique(), [ACCENT, COOL, WARM]):
        sub = sens[sens["parameter"] == parameter]
        axes[1].plot(sub["multiplier"], sub["ftl_share_recommended_pct"], marker="o",
                     color=colour, label=parameter.replace("_", " "))
    axes[1].axvline(1.0, color=INK, ls="--", lw=1)
    axes[1].legend(frameon=False, fontsize=9)
    _style(axes[1], "How much the answer depends on our invented rate card",
           xlabel="Assumption multiplier (1.0 = stated assumption)",
           ylabel="FTL share recommended (%)")
    fig.suptitle("FTL vs Carting: the recommendation and its fragility",
                 fontsize=15, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def prediction_scatter(predictions: pd.DataFrame, path: Path) -> Path:
    sample = predictions.sample(min(4000, len(predictions)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    limit = float(np.percentile(sample["actual_minutes"], 99))
    for ax, col, title in (
        (axes[0], "osrm_minutes", "OSRM estimate (incumbent)"),
        (axes[1], "pred_embedded", "Graph-enhanced model"),
    ):
        ax.scatter(sample["actual_minutes"], sample[col], s=6, alpha=0.25,
                   color=COOL if col != "pred_embedded" else ACCENT, edgecolors="none")
        ax.plot([0, limit], [0, limit], color=INK, ls="--", lw=1.2)
        ax.set_xlim(0, limit); ax.set_ylim(0, limit)
        _style(ax, title, xlabel="Actual (minutes)", ylabel="Predicted (minutes)")
    fig.suptitle("OSRM is biased low by construction; the model is centred on the line",
                 fontsize=15, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
