from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Palette validated
MARK = "#008268"
HIGHLIGHT = "#B36A05"
INK = "#131A19"
MUTED = "#566361"
RULE = "#DCE3E1"

# Groups called out in the write-up as the sharpest counter-examples.
CALLOUT = {"women", "islam"}


def _style(ax, xlabel: str, ylabel: str, title: str, subtitle: str) -> None:
    # Stacked in axes coordinates
    ax.text(0, 1.105, title, transform=ax.transAxes, fontsize=13,
            color=INK, va="bottom", fontweight="bold")
    ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=MUTED, va="bottom")
    ax.set_xlabel(xlabel, fontsize=10, color=MUTED, labelpad=9)
    ax.set_ylabel(ylabel, fontsize=10, color=MUTED, labelpad=9)
    ax.grid(True, color=RULE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def _scatter(ax, df: pd.DataFrame, xcol: str, rho: float,
             xlabel: str, title: str, subtitle: str) -> None:
    x = 100 * df[xcol].to_numpy(dtype=float)
    y = 100 * df["human_hate_share"].to_numpy(dtype=float)
    n = df["n_hostile"].to_numpy(dtype=float)
    # area by n, floored
    size = 40 + 320 * (n - n.min()) / max(1.0, n.max() - n.min())

    colours = [
        HIGHLIGHT if g in CALLOUT else MARK for g in df["group"]
    ]
    ax.scatter(x, y, s=size, c=colours, alpha=0.85, linewidths=1.4,
               edgecolors="white", zorder=3)

    for xi, yi, label in zip(x, y, df["group"]):
        colour = HIGHLIGHT if label in CALLOUT else MUTED
        weight = "semibold" if label in CALLOUT else "normal"
        ax.annotate(label, (xi, yi), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=8.5,
                    color=colour, fontweight="bold" if weight != "normal"
                    else "normal", zorder=4)

    _style(ax, xlabel, "human hate-share (%)", title, subtitle)
    ax.set_ylim(15, 92)
    pad = (x.max() - x.min()) * 0.14
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.text(
        0.985, 0.045, f"Spearman ρ = {rho:+.2f}",
        transform=ax.transAxes, ha="right", fontsize=10.5, color=INK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                  edgecolor=RULE, linewidth=0.9),
    )


def plot_confounds(
    slur_df: pd.DataFrame,
    agree_df: pd.DataFrame,
    rho_slur: float,
    rho_agree: float,
    out_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    _scatter(
        ax, slur_df, "slur_rate", rho_slur,
        "explicit slur presence among hostile posts (%)",
        "Severity grading does not track slur content",
        "each point is a target community; area is hostile-post count",
    )
    fig.tight_layout()
    p = out_dir / "confound_slur_presence.png"
    fig.savefig(p, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    _scatter(
        ax, agree_df, "hate_offensive_split", rho_agree,
        "hostile posts where annotators split hate / offensive (%)",
        "Severity grading is not explained by contested labelling",
        "each point is a target community; area is hostile-post count",
    )
    fig.tight_layout()
    p = out_dir / "confound_annotator_agreement.png"
    fig.savefig(p, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths


__all__ = ["plot_confounds"]
