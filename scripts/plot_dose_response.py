from __future__ import annotations

from pathlib import Path

import pandas as pd

# Palette validated
SPREAD_C = "#008268"
RHO_C = "#B36A05"
INK = "#131A19"
MUTED = "#566361"
RULE = "#DCE3E1"
EXCLUDED = "#9AA8A5"


def _verdict(df: pd.DataFrame) -> tuple[str, str]:
    usable = df[df["interpretable"]]
    trends = {}
    for col in ("spread_removed_pct", "rho_lost_pct"):
        v = usable[col].tolist()
        trends[col] = (
            "rising" if all(b >= a - 1e-9 for a, b in zip(v, v[1:]))
            else "falling" if all(b <= a + 1e-9 for a, b in zip(v, v[1:]))
            else "no trend"
        )
    if trends["spread_removed_pct"] == trends["rho_lost_pct"] == "rising":
        return ("Label-dependence rises with corpus exposure",
                "as predicted, on both measures")
    if "rising" in trends.values() and "falling" in trends.values():
        return ("The two measures of label-dependence disagree",
                "rank agreement and spread move in opposite directions — "
                "the prediction is not supported")
    return ("Corpus exposure does not predict label-dependence",
            f"spread: {trends['spread_removed_pct']}; "
            f"rank agreement: {trends['rho_lost_pct']}")


def plot_dose_response(df: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["DejaVu Sans"]

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(7.6, 6.6), dpi=200, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.18},
    )
    fig.patch.set_facecolor("white")
    for a in (ax, ax2):
        a.set_facecolor("white")

    x = range(len(df))
    labels = df["exposure"].tolist()
    usable = df["interpretable"].tolist()

    # top panel: the two label-dependence
    for col, colour, name in (
        ("spread_removed_pct", SPREAD_C, "group spread removed"),
        ("rho_lost_pct", RHO_C, "rank agreement lost"),
    ):
        vals = df[col].tolist()
        # solid through interpretable points, hollow markers for excluded ones
        ax.plot(x, vals, color=colour, linewidth=2, zorder=3, label=name)
        for xi, v, ok in zip(x, vals, usable):
            ax.scatter([xi], [v], s=90, zorder=4,
                       color=colour if ok else "white",
                       edgecolors=colour if ok else EXCLUDED, linewidths=1.8)
        ax.annotate(f"{vals[-1]:.0f}%", (list(x)[-1], vals[-1]),
                    textcoords="offset points", xytext=(10, -3),
                    color=colour, fontsize=10, fontweight="bold")

    ax.set_ylabel("label-dependence (%)", fontsize=10, color=MUTED, labelpad=9)
    title, subtitle = _verdict(df)
    ax.text(0, 1.115, title, transform=ax.transAxes, fontsize=13, color=INK,
            va="bottom", fontweight="bold")
    ax.text(0, 1.055, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=MUTED, va="bottom")
    ax.text(0, 1.005,
            "roberta-base throughout — same architecture, loop, folds and seed",
            transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.grid(True, color=RULE, linewidth=0.7)
    ax.set_axisbelow(True)

    # bottom panel: capability, so no
    ax2.bar(x, df["macro_f1"], color=[
        SPREAD_C if ok else EXCLUDED for ok in usable
    ], width=0.45, zorder=3)
    floor = 0.50
    ax2.axhline(floor, color=RHO_C, linewidth=1.4, linestyle="--", zorder=4)
    ax2.annotate("capability floor", (len(df) - 0.55, floor),
                 textcoords="offset points", xytext=(0, 5),
                 color=RHO_C, fontsize=8.5, ha="right")
    ax2.set_ylabel("macro-F1", fontsize=10, color=MUTED, labelpad=9)
    ax2.set_ylim(0, max(0.8, df["macro_f1"].max() * 1.15))
    ax2.grid(True, color=RULE, linewidth=0.7, axis="y")
    ax2.set_axisbelow(True)

    for a in (ax, ax2):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(RULE)
        a.tick_params(colors=MUTED, labelsize=9, length=0)

    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("share of what the model knows that came from this corpus",
                   fontsize=10, color=MUTED, labelpad=9)

    fig.tight_layout()
    path = out_dir / "dose_response.png"
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


__all__ = ["plot_dose_response"]
