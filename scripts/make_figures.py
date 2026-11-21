from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from src import config, crossval, data, figstyle
from src.figstyle import (
    FIG_WIDTH,
    GRID,
    HUMAN,
    INK,
    MUTED,
    RESULT_KEY,
    SYSTEM_LABELS,
    SYSTEM_STYLE,
    apply_style,
    category_axis,
    save,
    value_grid,
)
from src.groups import directional_table, wilson

SYSTEMS = list(SYSTEM_STYLE)


# data


def load_hate_share() -> pd.DataFrame:
    df = pd.read_csv(config.RESULTS_DIR / "group_hate_share.csv")
    return df.sort_values("human", ascending=False).reset_index(drop=True)


def load_escalation() -> pd.DataFrame:
    task = config.DEFAULT_TASK
    corpus = data.labelled_hatexplain(task=task).sort_values(
        "post_id", kind="mergesort"
    ).reset_index(drop=True)

    frames = []
    for key in SYSTEMS:
        blob = crossval.load_oof(RESULT_KEY[key])
        seed = blob["seeds"][0]
        preds = np.asarray(blob["predictions"][str(seed)], dtype=int)
        tbl = directional_table(corpus, preds, task, config.ANALYSIS_GROUPS)
        tbl.insert(0, "system", key)
        tbl.insert(1, "seed", seed)
        frames.append(tbl)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(config.RESULTS_DIR / "group_escalation.csv", index=False)
    return out


def _pct_axis(ax, axis: str = "x") -> None:
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _: f"{100 * v:.0f}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


# 1. the gradient


def fig_gradient(shares):
    import matplotlib.pyplot as plt

    order = shares["group"].tolist()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 5.4))

    series = [("human", *HUMAN)] + [(k, *SYSTEM_STYLE[k]) for k in SYSTEMS]
    tbl = shares.set_index("group")
    values = {
        key: tbl.loc[order, "human" if key == "human" else RESULT_KEY[key]]
        for key, _, _ in series
    }
    y = np.arange(len(order))[::-1]

    # Dodge, or the SVM disappears
    for i in range(0, len(order), 2):
        ax.axhspan(y[i] - 0.5, y[i] + 0.5, color="#f4f6f8", zorder=0,
                   linewidth=0)

    offsets = np.linspace(-0.28, 0.28, len(series))
    for (key, colour, marker), dy in zip(series, offsets):
        human_row = key == "human"
        # A star reads smaller
        ax.scatter(values[key], y + dy, s=72 if human_row else 26,
                   marker=marker, color=colour, edgecolors="white",
                   linewidths=0.6 if human_row else 0.5,
                   zorder=6 if human_row else 4,
                   label="human annotators" if human_row else SYSTEM_LABELS[key],
                   clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.set_xlabel("share of hostile posts graded hate speech (%)")
    _pct_axis(ax)
    value_grid(ax, "x")
    category_axis(ax, "y")
    # Legend below the axes
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.105), ncol=3,
              columnspacing=1.0, handletextpad=0.25,
              fontsize=figstyle.MIN_PT - 0.5)
    return fig


# 2. the confound scatter


def fig_confound():
    import matplotlib.pyplot as plt

    df = pd.read_csv(config.RESULTS_DIR / "group_slur_presence.csv")
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.3))

    x = 100 * df["slur_rate"].to_numpy()
    y = 100 * df["human_hate_share"].to_numpy()

    # Three communities carry it
    carry = {"african", "homosexual", "women"}

    # Vertical, so not read as a fit
    band = df[df["group"].isin(carry)]["slur_rate"]
    ax.axvspan(100 * band.min() - 0.5, 100 * band.max() + 0.5,
               color="#eef1f4", zorder=0, linewidth=0)

    ax.scatter(x, y, s=30, marker="o", color=HUMAN[0], edgecolors="white",
               linewidths=0.6, zorder=4)
    placement = {
        "african": (0, 8), "homosexual": (0, -13), "women": (0, 8),
        "jewish": (0, 7), "hispanic": (-4, 7), "arab": (0, -13),
        "islam": (0, 7), "asian": (0, 7), "refugee": (0, 7),
        "caucasian": (0, -13),
    }
    for xi, yi, name in zip(x, y, df["group"]):
        dx, dy = placement.get(name, (0, 7))
        # Ink, not weight: no bold Roboto
        ax.annotate(
            name, (xi, yi), textcoords="offset points", xytext=(dx, dy),
            ha="center", fontsize=figstyle.MIN_PT - 0.5,
            color=INK if name in carry else MUTED, zorder=5,
        )

    ax.set_xlabel("hostile posts containing an explicit slur (%)")
    ax.set_ylabel("hostile posts graded hate speech (%)")
    value_grid(ax, "x")
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_xlim(55, 95)
    ax.set_ylim(20, 88)
    return fig


# 3. performance against cost


def fig_cost():
    import matplotlib.pyplot as plt

    rows = []
    for key in SYSTEMS:
        if key == "llm_zero_shot":
            continue
        blob = json.loads(
            (config.RESULTS_DIR / f"{RESULT_KEY[key]}.json").read_text()
        )
        rows.append({
            "key": key,
            "train_s": blob["cost"]["train_seconds"],
            "macro_f1": blob["headline"]["macro_f1"],
        })
    df = pd.DataFrame(rows)

    # No training, no x position
    llm_f1 = _llm_macro_f1()

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.1))
    ax.axhline(llm_f1, color=SYSTEM_STYLE["llm_zero_shot"][0], linewidth=1.1,
               linestyle=(0, (4, 2)), zorder=2)
    # Ink, not the series colour
    ax.annotate(
        "LLM (zero-shot) — no training", xy=(0.0, llm_f1),
        xycoords=("axes fraction", "data"), textcoords="offset points",
        xytext=(1, 5), ha="left", fontsize=figstyle.MIN_PT - 0.5,
        color=INK, zorder=6,
    )

    # NB sits on the LLM line
    offsets = {
        "rule_based_lexicon": (7, -3), "naive_bayes": (7, -12),
        "tfidf_linear_svc": (7, -3), "roberta_base": (-6, -13),
    }
    for _, r in df.iterrows():
        colour, marker = SYSTEM_STYLE[r["key"]]
        ax.scatter(r["train_s"], r["macro_f1"], s=42, marker=marker,
                   color=colour, edgecolors="white", linewidths=0.6, zorder=5)
        dx, dy = offsets[r["key"]]
        ax.annotate(
            SYSTEM_LABELS[r["key"]], (r["train_s"], r["macro_f1"]),
            textcoords="offset points", xytext=(dx, dy),
            ha="right" if dx < 0 else "left",
            fontsize=figstyle.MIN_PT - 0.5, color=INK, zorder=6,
        )

    ax.set_xscale("log")
    ax.tick_params(axis="x", which="minor", length=1.2, width=0.5)
    ax.set_xlabel("training wall-clock, seconds (log scale)")
    ax.set_ylabel("macro-F1")
    value_grid(ax, "x")
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_ylim(0.38, 0.74)
    ax.set_xlim(0.05, 900)
    return fig


def _llm_macro_f1() -> float:
    from sklearn.metrics import f1_score

    task = config.DEFAULT_TASK
    corpus = data.labelled_hatexplain(task=task).sort_values(
        "post_id", kind="mergesort"
    ).reset_index(drop=True)
    blob = crossval.load_oof(RESULT_KEY["llm_zero_shot"])
    preds = np.asarray(blob["predictions"][str(blob["seeds"][0])], dtype=int)

    splits = json.loads(
        (config.PROCESSED_DIR / config.DEFAULT_SPLIT.filename).read_text(
            encoding="utf-8"
        )
    )
    held_out = set(map(str, splits["test"]))
    mask = corpus["post_id"].astype(str).isin(held_out).to_numpy()
    return float(f1_score(corpus["label"][mask], preds[mask], average="macro",
                          zero_division=0))


# 4. escalation by community


def fig_escalation(esc):
    import matplotlib.pyplot as plt

    shown = ["rule_based_lexicon", "tfidf_linear_svc", "roberta_base"]
    by_system = {k: g.set_index("group") for k, g in esc.groupby("system")}
    order = (by_system["roberta_base"]["escalation"]
             .sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 4.4))
    offsets = np.linspace(-0.26, 0.26, len(shown))

    for key, dy in zip(shown, offsets):
        tbl = by_system[key]
        colour, marker = SYSTEM_STYLE[key]
        xs, los, his, ys = [], [], [], []
        for i, g in enumerate(order):
            row = tbl.loc[g]
            n = int(row["n_human_offensive"])
            p = float(row["escalation"])
            ci = wilson(int(round(p * n)), n)
            xs.append(100 * p)
            los.append(100 * (p - ci.lo))
            his.append(100 * (ci.hi - p))
            ys.append(len(order) - 1 - i + dy)
        ax.errorbar(xs, ys, xerr=[los, his], fmt="none", ecolor=colour,
                    elinewidth=0.9, capsize=1.6, capthick=0.7, zorder=3,
                    alpha=0.85)
        ax.scatter(xs, ys, s=26, marker=marker, color=colour,
                   edgecolors="white", linewidths=0.5, zorder=4,
                   label=SYSTEM_LABELS[key])

    ax.set_yticks(np.arange(len(order))[::-1])
    ax.set_yticklabels(order)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlabel("human-offensive posts graded hate speech (%)")
    value_grid(ax, "x")
    category_axis(ax, "y")
    ax.legend(loc="lower right", borderaxespad=0.1,
              fontsize=figstyle.MIN_PT - 0.5)
    return fig


# 5. label-dependence, two panels


def fig_label_dependence():
    import matplotlib.pyplot as plt

    across = []
    for key in ("naive_bayes", "tfidf_linear_svc", "roberta_base"):
        blob = json.loads(
            (config.RESULTS_DIR / f"label_correction_{RESULT_KEY[key]}.json")
            .read_text()
        )
        across.append({
            "key": key,
            "spread": 100 * blob["spread"]["fraction_of_model_spread_removed"],
        })
    across.append({"key": "llm_zero_shot", "spread": 0.0})

    dose = pd.read_csv(config.RESULTS_DIR / "dose_response.csv")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH * 1.62, 2.9),
        gridspec_kw={"width_ratios": [1, 1.12], "wspace": 0.34},
    )

    # left: across systems
    xs = np.arange(len(across))
    for i, r in enumerate(across):
        colour, marker = SYSTEM_STYLE[r["key"]]
        ax1.plot([i, i], [0, r["spread"]], color=colour, linewidth=1.1,
                 zorder=3, solid_capstyle="butt")
        ax1.scatter([i], [r["spread"]], s=44, marker=marker, color=colour,
                    edgecolors="white", linewidths=0.6, zorder=5)
        ax1.annotate(f"{r['spread']:.0f}", (i, r["spread"]),
                     textcoords="offset points", xytext=(0, 6), ha="center",
                     fontsize=figstyle.MIN_PT - 0.5, color=INK)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([SYSTEM_LABELS[r["key"]] for r in across],
                        rotation=28, ha="right")
    ax1.set_ylabel("group spread removed by\ncorrecting the labels (%)")
    # Room below zero for the 0%
    ax1.set_ylim(-1.4, 44)
    ax1.set_xlim(-0.6, len(across) - 0.4)
    ax1.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax1.grid(False, axis="x")
    ax1.set_axisbelow(True)
    category_axis(ax1, "x")

    # Within one architecture
    x = np.arange(len(dose))
    for col, marker, dash, name, colour in (
        ("spread_removed_pct", "D", "solid", "group spread removed",
         SYSTEM_STYLE["roberta_base"][0]),
        ("rho_lost_pct", "o", (0, (4, 2)), "rank agreement lost", HUMAN[0]),
    ):
        ax2.plot(x, dose[col], color=colour, linewidth=1.2, linestyle=dash,
                 zorder=3)
        ax2.scatter(x, dose[col], s=34, marker=marker, color=colour,
                    edgecolors="white", linewidths=0.6, zorder=5, label=name)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dose["exposure"])
    ax2.set_xlabel("RoBERTa: share of training corpus seen")
    ax2.set_ylabel("label-dependence (%)")
    ax2.set_ylim(-1.5, 20)
    ax2.set_xlim(-0.35, len(dose) - 0.65)
    ax2.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax2.grid(False, axis="x")
    ax2.set_axisbelow(True)
    category_axis(ax2, "x")
    ax2.legend(loc="center right", borderaxespad=0.2,
               fontsize=figstyle.MIN_PT - 0.5)
    return fig


CAPTIONS = {
    "gradient": (
        "Share of hostile posts graded as hate speech, by target community "
        "and rater. Communities are ordered by the human rate, which spans "
        "48.6 points. All four learned systems reproduce that ordering "
        "(Spearman's $\\rho$ from $+0.903$ to $+0.988$) while grading more "
        "severely than the annotators did in 38 of 40 system--community "
        "cells; the two exceptions are RoBERTa on refugee (28.0\\% against "
        "32.8\\%) and on women (28.9\\% against 29.9\\%). The rule-based "
        "lexicon has no representation of severity and stays between 46.4\\% "
        "and 55.6\\% for nine of the ten communities, the exception being "
        "Arab at 71.2\\%. Out-of-fold predictions over all 19{,}229 posts; "
        "RoBERTa is the mean of three seeds."
    ),
    "confound": (
        "Human severity grading against explicit slur content, by target "
        "community. No trend line is fitted: Spearman's $\\rho = +0.139$ "
        "($p = 0.701$). African, homosexual and women carry slur rates within "
        "3.1 points of one another while their hate-shares span 45.8 points, "
        "which rules out content severity as the explanation for the gradient."
    ),
    "cost": (
        "Held-out macro-F1 against training wall-clock (log scale). RoBERTa "
        "buys 4.8 macro-F1 points over the TF-IDF SVM for 73$\\times$ the "
        "training time. The prompted LLM does not train at all and so has no "
        "position on the time axis; it is shown as a horizontal reference at "
        "its own macro-F1. All five scores are on the same held-out split of "
        "2,883 posts."
    ),
    "escalation": (
        "Escalation rate --- the share of posts human annotators called "
        "\\emph{offensive} that the system called \\emph{hate speech} --- by "
        "target community, with Wilson 95\\% intervals. Communities are "
        "ordered by RoBERTa's rate, which runs from 56.6\\% for Hispanic "
        "targets to 7.9\\% for women, a factor of 7.2 with non-overlapping "
        "intervals. The SVM tracks that ordering closely ($\\rho = +0.952$); "
        "the rule-based lexicon does not track it at all ($\\rho = +0.055$) "
        "and spans only a factor of 1.9, escalating on 37.2\\% of "
        "human-offensive posts about women where RoBERTa escalates on 7.9\\%."
    ),
    "label_dependence": (
        "Left: the share of each system's group ordering removed by "
        "correcting the corpus labels, across four architectures. Right: the "
        "same intervention applied within a single architecture "
        "(\\texttt{roberta-base} throughout) at four levels of corpus "
        "exposure. Both series in the right-hand panel are percentages on one "
        "shared axis; note that the two panels use different vertical ranges. "
        "The two measures of label-dependence move in opposite directions on "
        "the right, so the corpus-exposure account is not supported by the "
        "within-architecture test."
    ),
}

BUILDERS = {
    "gradient": "group",
    "confound": "plain",
    "cost": "plain",
    "escalation": "group",
    "label_dependence": "plain",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=list(BUILDERS))
    args = parser.parse_args()

    apply_style()
    import matplotlib.pyplot as plt

    shares, esc = load_hate_share(), load_escalation()
    made = {}
    for name in args.only:
        fig = {
            "gradient": lambda: fig_gradient(shares),
            "confound": fig_confound,
            "cost": fig_cost,
            "escalation": lambda: fig_escalation(esc),
            "label_dependence": fig_label_dependence,
        }[name]()
        paths = save(fig, name)
        plt.close(fig)
        made[name] = paths
        size = fig.get_size_inches()
        print(f"{name:18s} {size[0]:.2f} x {size[1]:.2f} in -> {paths['pdf'].name}")

    cap = figstyle.FIGURE_DIR / "captions.md"
    cap.write_text(
        "# Figure captions\n\nPaste into the .tex; the figures carry no "
        "titles of their own.\n\n"
        + "\n\n".join(
            f"## {n}\n\n```latex\n\\caption{{{CAPTIONS[n]}}}\n```"
            for n in BUILDERS
        ) + "\n",
        encoding="utf-8",
    )
    print(f"\ncaptions -> {cap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
