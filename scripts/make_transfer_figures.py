from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src import config, figstyle
from src.figstyle import (
    FIG_WIDTH, GRID, HUMAN, INK, MUTED, SYSTEM_LABELS, SYSTEM_STYLE,
    apply_style, category_axis, save, value_grid,
)

# LLM scored on the held-out rows
LLM_HELDOUT_MACRO_F1 = 0.6051

ORDER = ["rule_based_lexicon", "naive_bayes", "tfidf_linear_svc",
         "tfidf_logistic_regression", "roberta_base", "llm_zero_shot"]

# RoBERTa seed spread
ROBERTA_SD = {"a": 0.200, "b": 0.211, "c": 0.209}


def _slope(ax, rows, left_label, right_label, fmt="{:.3f}", pad=0.055):
    for key, colour, marker, label, l, r in rows:
        ax.plot([0, 1], [l, r], color=colour, linewidth=1.3, zorder=3,
                solid_capstyle="round")
        ax.scatter([0, 1], [l, r], s=34, marker=marker, color=colour,
                   edgecolors="white", linewidths=0.6, zorder=5)
    ax.set_xlim(-pad * 7, 1 + pad * 7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([left_label, right_label])
    category_axis(ax, "x")
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _dodge(values, min_gap):
    idx = np.argsort(values)
    out = np.array(values, dtype=float)
    for i in range(1, len(idx)):
        a, b = idx[i - 1], idx[i]
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def fig_inversion():
    import json
    import matplotlib.pyplot as plt

    mhs = pd.read_csv(config.RESULTS_DIR / "mhs_transfer_performance.csv")
    log = pd.read_csv(config.RESULTS_DIR / "mhs_logreg_performance.csv")
    mhs = pd.concat([mhs, log], ignore_index=True)
    mhs = mhs[mhs["mapping"] == "bands"].set_index("system")["macro_f1"]

    hx = {}
    for key in ORDER:
        if key == "llm_zero_shot":
            hx[key] = LLM_HELDOUT_MACRO_F1
            continue
        blob = json.loads(
            (config.RESULTS_DIR / f"{key}.json").read_text(encoding="utf-8"))
        hx[key] = blob["headline"]["macro_f1"]

    key_for = {"llm_zero_shot": "llm_zero_shot_haiku_corpus"}
    rows = [(k, *SYSTEM_STYLE[k], SYSTEM_LABELS[k], hx[k],
             float(mhs[key_for.get(k, k)])) for k in ORDER]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.12, 3.5))
    _slope(ax, rows, "HateXplain\n(held-out)", "Measuring Hate Speech\n(transfer)")

    lefts = _dodge([r[4] for r in rows], 0.0135)
    rights = _dodge([r[5] for r in rows], 0.0135)
    for (key, colour, marker, label, l, r), ly, ry in zip(rows, lefts, rights):
        ax.annotate(f"{label}  {l:.3f}", (0, ly), textcoords="offset points",
                    xytext=(-7, -3), ha="right", fontsize=figstyle.MIN_PT - 1.2,
                    color=INK)
        ax.annotate(f"{r:.3f}  {label}", (1, ry), textcoords="offset points",
                    xytext=(7, -3), ha="left", fontsize=figstyle.MIN_PT - 1.2,
                    color=INK)
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0.39, 0.70)
    # Endpoints carry their own values
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    return fig


def fig_replication():
    import matplotlib.pyplot as plt

    d = pd.read_csv(config.RESULTS_DIR / "mhs_irt_check.csv")
    d = d.sort_values("hatexplain", ascending=False)
    hx_spread = 100 * (d["hatexplain"].max() - d["hatexplain"].min())
    mhs_spread = 100 * (d["MHS IRT"].max() - d["MHS IRT"].min())

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.12, 4.9))
    rows = [(g, HUMAN[0], "o", g, l, r) for g, l, r in
            zip(d["community"], d["hatexplain"], d["MHS IRT"])]
    # Two communities carry the argument; everything else is context.
    carry = {"jewish", "women"}
    for g, _, marker, _, l, r in rows:
        emph = g in carry
        ax.plot([0, 1], [l, r], color=INK if emph else "#b9bec4",
                linewidth=1.6 if emph else 1.0, zorder=5 if emph else 3)
        ax.scatter([0, 1], [l, r], s=30 if emph else 20, marker="o",
                   color=INK if emph else "#b9bec4", edgecolors="white",
                   linewidths=0.6, zorder=6 if emph else 4)
    _slope(ax, [], "HateXplain\nannotators", "MHS\nannotators")

    lefts = _dodge(d["hatexplain"].tolist(), 0.024)
    rights = _dodge(d["MHS IRT"].tolist(), 0.024)
    for (g, _, _, _, l, r), ly, ry in zip(rows, lefts, rights):
        emph = g in carry
        col = INK if emph else MUTED
        ax.annotate(f"{g}  {100*l:.1f}", (0, ly), textcoords="offset points",
                    xytext=(-7, -3), ha="right",
                    fontsize=figstyle.MIN_PT - 1.2, color=col)
        ax.annotate(f"{100*r:.1f}  {g}", (1, ry), textcoords="offset points",
                    xytext=(7, -3), ha="left",
                    fontsize=figstyle.MIN_PT - 1.2, color=col)

    ax.set_ylabel("share of hostile content graded hate speech (%)")
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}"))
    ax.set_ylim(0.24, 0.85)
    # Endpoints carry their own values
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.annotate(f"spread {hx_spread:.1f} pts", (0, 0.255), ha="center",
                fontsize=figstyle.MIN_PT - 1, color=MUTED)
    ax.annotate(f"spread {mhs_spread:.1f} pts", (1, 0.255), ha="center",
                fontsize=figstyle.MIN_PT - 1, color=MUTED)
    return fig


def fig_carried():
    import matplotlib.pyplot as plt

    d = pd.read_csv(config.RESULTS_DIR / "mhs_transfer_correlations.csv")
    d = d.set_index("system")
    # logreg was run separately; values from scripts.mhs_logreg
    d.loc["tfidf_logistic_regression"] = {
        "a_vs_hx_human": 0.442, "a_p": 0.200, "b_vs_mhs_human": -0.297,
        "b_p": 0.405, "c_vs_own_hx": 0.418, "c_p": 0.229}

    key_for = {"llm_zero_shot": "llm_zero_shot_haiku_corpus"}
    series = (("a_vs_hx_human", "o", INK, "vs HateXplain humans"),
              ("b_vs_mhs_human", "s", "#c0392b", "vs MHS humans"),
              ("c_vs_own_hx", "^", "#7f8c8d", "vs own HateXplain ordering"))
    sd_for = {"a_vs_hx_human": ROBERTA_SD["a"],
              "b_vs_mhs_human": ROBERTA_SD["b"],
              "c_vs_own_hx": ROBERTA_SD["c"]}

    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 1.32, 3.4))
    rows = list(reversed(ORDER))
    y = np.arange(len(rows))
    offs = np.linspace(-0.24, 0.24, len(series))

    ax.axvline(0, color="#9aa0a6", linewidth=0.9, zorder=2)
    for i, key in enumerate(rows):
        if key == "rule_based_lexicon":
            ax.axhspan(i - 0.45, i + 0.45, color="#f2f4f6", zorder=0, linewidth=0)
    for (col, marker, colour, label), dy in zip(series, offs):
        vals = [d.loc[key_for.get(k, k), col] for k in rows]
        ax.scatter(vals, y + dy, s=30, marker=marker, color=colour,
                   edgecolors="white", linewidths=0.6, zorder=5, label=label)
        ri = rows.index("roberta_base")
        ax.errorbar([vals[ri]], [ri + dy], xerr=[[sd_for[col]], [sd_for[col]]],
                    fmt="none", ecolor=colour, elinewidth=0.9, capsize=1.8,
                    capthick=0.7, zorder=4, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels([SYSTEM_LABELS[k] + ("  (control)" if
                        k == "rule_based_lexicon" else "") for k in rows])
    ax.set_ylim(-0.55, len(rows) - 0.45)
    # Literal rho, not mathtext
    ax.set_xlabel("Spearman ρ with the ten-community ordering")
    ax.set_xlim(-0.72, 1.0)
    value_grid(ax, "x")
    category_axis(ax, "y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
              columnspacing=1.1, handletextpad=0.25,
              fontsize=figstyle.MIN_PT - 1.2)
    return fig


def fig_contestedness():
    import matplotlib.pyplot as plt

    d = pd.read_csv(config.RESULTS_DIR / "group_annotator_agreement.csv")
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.3))
    x = 100 * d["hate_offensive_split"].to_numpy()
    y = 100 * d["human_hate_share"].to_numpy()

    carry = {"islam", "arab", "hispanic"}
    ax.scatter(x, y, s=30, marker="o", color=HUMAN[0], edgecolors="white",
               linewidths=0.6, zorder=4)
    placement = {"jewish": (0, 7), "african": (0, 7), "hispanic": (0, -13),
                 "arab": (6, 4), "islam": (0, -13), "asian": (0, 7),
                 "homosexual": (0, 7), "refugee": (0, 7), "women": (0, 7),
                 "caucasian": (0, -13)}
    for xi, yi, name in zip(x, y, d["group"]):
        dx, dy = placement.get(name, (0, 7))
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(dx, dy),
                    ha="center", fontsize=figstyle.MIN_PT - 0.5,
                    color=INK if name in carry else MUTED, zorder=5)
    ax.set_xlabel("hostile posts where annotators split hate/offensive (%)")
    ax.set_ylabel("hostile posts graded hate speech (%)")
    value_grid(ax, "x")
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_xlim(28, 51)
    ax.set_ylim(20, 88)
    return fig


CAPTIONS = {
    "inversion": (
        "Macro-F1 on the HateXplain held-out split against macro-F1 on "
        "Measuring Hate Speech, for the same six systems run as inference "
        "only. A single shared vertical scale, so a crossing means the two "
        "systems genuinely change places. The rule-based lexicon is the only "
        "system that improves (0.410 to 0.499); both linear variants fall "
        "past it (SVM 0.631 to 0.480, logistic regression 0.640 to 0.485), as "
        "does Naive Bayes (0.608 to 0.487). The prompted LLM, which never saw "
        "HateXplain's labels, moves least (0.605 to 0.581) and finishes "
        "first. MHS scores use the `bands` gold mapping; the ordering is the "
        "same under the other two mappings tested."
    ),
    "replication": (
        "Human severity grading by target community on two corpora: "
        "HateXplain's three annotators per post against Measuring Hate "
        "Speech's 7,912 annotators, on the same ten communities under the "
        "target mapping given in the text. Shared vertical scale. The "
        "ordering does not replicate --- Spearman's $\\rho = -0.19$ --- and "
        "the spread contracts from 48.6 points to 18.7. Jewish falls from "
        "first to ninth and women rise from ninth to third; both are drawn "
        "in full ink."
    ),
    "carried": (
        "What each system's community ordering on MHS text correlates with: "
        "the HateXplain annotators, the MHS annotators, or that system's own "
        "ordering on HateXplain. The middle series is lowest for every system "
        "and negative for three, so models reading MHS text order communities "
        "more like the annotators of the corpus they were trained on than "
        "like the annotators who labelled the text in front of them. The "
        "rule-based lexicon is shaded as a control: it has no learned "
        "ordering, and correspondingly carries nothing. RoBERTa's bars are "
        "$\\pm$1 sd over three seeds; the other systems are deterministic."
    ),
    "contestedness": (
        "Human severity grading against annotator contestedness --- the share "
        "of a community's hostile posts on which at least one annotator said "
        "\\emph{hate speech} and at least one said \\emph{offensive}. No "
        "trend line is fitted: Spearman's $\\rho = +0.236$ ($p = 0.511$). "
        "Islam (44.8\\%) and Arab (47.9\\%) are both above average in "
        "contestedness, but Hispanic sits at 44.5\\% with a hate-share 9.4 "
        "points higher than Islam's. Companion to the slur-content figure; "
        "neither confound accounts for the gradient."
    ),
}

BUILDERS = {"inversion": fig_inversion, "replication": fig_replication,
            "carried": fig_carried, "contestedness": fig_contestedness}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=list(BUILDERS))
    args = ap.parse_args()

    apply_style()
    import matplotlib.pyplot as plt

    for name in args.only:
        fig = BUILDERS[name]()
        paths = save(fig, name)
        size = fig.get_size_inches()
        plt.close(fig)
        print(f"{name:16s} {size[0]:.2f} x {size[1]:.2f} in -> {paths['pdf'].name}")

    cap = figstyle.FIGURE_DIR / "captions_transfer.md"
    cap.write_text(
        "# Captions for the transfer figures\n\n"
        + "\n\n".join(f"## {n}\n\n```latex\n\\caption{{{CAPTIONS[n]}}}\n```"
                      for n in BUILDERS) + "\n", encoding="utf-8")
    print(f"\ncaptions -> {cap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
