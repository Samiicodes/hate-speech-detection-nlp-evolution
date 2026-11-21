from __future__ import annotations

from pathlib import Path

from .config import RESULTS_DIR

FIGURE_DIR: Path = RESULTS_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# the palette

# system key to colour and marker
SYSTEM_STYLE: dict[str, tuple[str, str]] = {
    "rule_based_lexicon": ("#2a78d6", "o"),   # blue,    circle
    "naive_bayes":        ("#eb6834", "s"),   # orange,  square
    "tfidf_linear_svc":   ("#1baf7a", "^"),   # aqua,    triangle
    # System 3's other variant
    "tfidf_logistic_regression": ("#7b5aa6", "X"),  # purple, filled x
    "roberta_base":       ("#eda100", "D"),   # yellow,  diamond
    "llm_zero_shot":      ("#e87ba4", "P"),   # magenta, filled plus
}
# Plus, not down-triangle: greyscale

# Human baseline, not a system
HUMAN: tuple[str, str] = ("#333333", "*")

# Display names. Short enough to direct-label without wrapping.
SYSTEM_LABELS: dict[str, str] = {
    "rule_based_lexicon": "lexicon",
    "naive_bayes": "Naive Bayes",
    "tfidf_linear_svc": "TF-IDF + SVM",
    "tfidf_logistic_regression": "TF-IDF + LogReg",
    "roberta_base": "RoBERTa",
    "llm_zero_shot": "LLM (zero-shot)",
}

# style key to results filename
RESULT_KEY: dict[str, str] = {
    "rule_based_lexicon": "rule_based_lexicon",
    "naive_bayes": "naive_bayes",
    "tfidf_linear_svc": "tfidf_linear_svc",
    "tfidf_logistic_regression": "tfidf_logistic_regression",
    "roberta_base": "roberta_base",
    "llm_zero_shot": "llm_zero_shot_haiku_corpus",
}

# ink

INK = "#1a1a1a"        # data and emphasis: the darkest thing on the page
LABEL = "#3d3d3d"      # axis labels
MUTED = "#6b6b6b"      # tick labels, annotations
GRID = "#d9d9d9"       # recessive
SPINE = "#b0b0b0"

# about 60% of A4 text width
FIG_WIDTH = 3.8

# minimum point size, set for paper
MIN_PT = 9.0


def apply_style() -> None:
    import matplotlib
    from matplotlib import rcParams

    matplotlib.use("Agg")

    # Roboto, not DejaVu
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = ["Roboto", "Arial", "Segoe UI", "sans-serif"]

    rcParams["font.size"] = MIN_PT
    rcParams["axes.labelsize"] = MIN_PT + 0.5
    rcParams["axes.titlesize"] = MIN_PT + 1
    rcParams["xtick.labelsize"] = MIN_PT
    rcParams["ytick.labelsize"] = MIN_PT
    rcParams["legend.fontsize"] = MIN_PT

    # white, never grey
    rcParams["figure.facecolor"] = "white"
    rcParams["axes.facecolor"] = "white"
    rcParams["savefig.facecolor"] = "white"

    # left and bottom spines only
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["axes.edgecolor"] = SPINE
    rcParams["axes.linewidth"] = 0.7
    rcParams["axes.labelcolor"] = LABEL

    # recessive grid, behind the data
    rcParams["grid.color"] = GRID
    rcParams["grid.linewidth"] = 0.6
    rcParams["axes.axisbelow"] = True

    # short outward ticks
    rcParams["xtick.direction"] = "out"
    rcParams["ytick.direction"] = "out"
    rcParams["xtick.major.size"] = 2.5
    rcParams["ytick.major.size"] = 2.5
    rcParams["xtick.major.width"] = 0.7
    rcParams["ytick.major.width"] = 0.7
    rcParams["xtick.color"] = MUTED
    rcParams["ytick.color"] = MUTED

    # unframed legend
    rcParams["legend.frameon"] = False
    rcParams["legend.handletextpad"] = 0.45
    rcParams["legend.labelspacing"] = 0.35
    rcParams["legend.borderpad"] = 0.2

    # vector output with real text, not outlines
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["savefig.bbox"] = "tight"
    rcParams["savefig.pad_inches"] = 0.02


def value_grid(ax, axis: str = "x") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6, zorder=0)
    ax.grid(False, axis="y" if axis == "x" else "x")
    ax.set_axisbelow(True)


def category_axis(ax, axis: str = "y") -> None:
    ax.tick_params(axis=axis, length=0)


def save(fig, name: str, greyscale: bool = True) -> dict[str, Path]:
    out: dict[str, Path] = {}
    pdf = FIGURE_DIR / f"{name}.pdf"
    fig.savefig(pdf)
    out["pdf"] = pdf

    png = FIGURE_DIR / f"{name}.png"
    fig.savefig(png, dpi=300)
    out["png"] = png

    if greyscale:
        from PIL import Image

        grey = FIGURE_DIR / f"{name}_grey.png"
        Image.open(png).convert("L").save(grey)
        out["greyscale"] = grey
    return out


__all__ = [
    "FIGURE_DIR",
    "FIG_WIDTH",
    "GRID",
    "HUMAN",
    "INK",
    "LABEL",
    "MIN_PT",
    "MUTED",
    "RESULT_KEY",
    "SYSTEM_LABELS",
    "SYSTEM_STYLE",
    "apply_style",
    "category_axis",
    "save",
    "value_grid",
]
