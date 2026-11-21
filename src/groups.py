from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    ANALYSIS_GROUPS,
    HOSTILE_LABELS,
    MIN_GROUP_N,
    TaskConfig,
)


# interval


@dataclass(frozen=True)
class Proportion:

    k: int
    n: int
    point: float | None
    lo: float | None
    hi: float | None

    @property
    def reliable(self) -> bool:
        return self.n >= MIN_GROUP_N

    def as_dict(self) -> dict:
        return {
            "k": self.k, "n": self.n, "point": self.point,
            "lo": self.lo, "hi": self.hi, "reliable": self.reliable,
        }

    def __str__(self) -> str:
        if self.point is None:
            return f"n/a (n={self.n})"
        return (
            f"{100 * self.point:.1f}% [{100 * self.lo:.1f}-{100 * self.hi:.1f}] "
            f"n={self.n}"
        )


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Proportion:
    if n == 0:
        return Proportion(k=0, n=0, point=None, lo=None, hi=None)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Proportion(
        k=k, n=n, point=p,
        lo=max(0.0, centre - half), hi=min(1.0, centre + half),
    )


# group membership


def group_mask(df: pd.DataFrame, group: str) -> np.ndarray:
    return df["target_groups"].map(lambda gs: group in gs).to_numpy(dtype=bool)


def available_groups(df: pd.DataFrame, min_n: int = 1) -> list[str]:
    hostile = df[df["majority_label"].isin(HOSTILE_LABELS)]
    counts: dict[str, int] = {}
    for groups in hostile["target_groups"]:
        for g in groups:
            counts[g] = counts.get(g, 0) + 1
    return [g for g, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= min_n]


# the human baseline


def human_hate_share(df: pd.DataFrame, group: str) -> Proportion:
    sub = df[group_mask(df, group)]
    hostile = sub[sub["majority_label"].isin(HOSTILE_LABELS)]
    return wilson(int((hostile["majority_label"] == "hatespeech").sum()), len(hostile))


def human_baseline_table(
    df: pd.DataFrame, groups: tuple[str, ...] = ANALYSIS_GROUPS
) -> pd.DataFrame:
    rows = []
    for group in groups:
        p = human_hate_share(df, group)
        rows.append({
            "group": group,
            "n_hostile": p.n,
            "n_hate": p.k,
            "hate_share": p.point,
            "ci_lo": p.lo,
            "ci_hi": p.hi,
            "reliable": p.reliable,
        })
    return pd.DataFrame(rows).sort_values(
        "hate_share", ascending=False, na_position="last"
    ).reset_index(drop=True)


# model-side measures


def model_group_metrics(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    group: str,
    task: TaskConfig,
) -> dict:
    hate_idx = task.index_of("hatespeech")
    off_idx = task.index_of("offensive")

    mask = group_mask(df, group)
    sub_labels = df.loc[mask, "majority_label"].to_numpy()
    sub_pred = np.asarray(y_pred)[mask]

    human_hostile = np.isin(sub_labels, HOSTILE_LABELS)
    human_hate = sub_labels == "hatespeech"
    human_off = sub_labels == "offensive"

    pred_hate = sub_pred == hate_idx
    pred_off = sub_pred == off_idx
    pred_hostile = pred_hate | pred_off

    # a. escalation
    escalation = wilson(int((human_off & pred_hate).sum()), int(human_off.sum()))
    # b. de-escalation
    de_escalation = wilson(int((human_hate & pred_off).sum()), int(human_hate.sum()))
    # c. model hate-share
    model_share = wilson(int(pred_hate.sum()), int(pred_hostile.sum()))
    # human measure, same rows
    human_share = wilson(int(human_hate.sum()), int(human_hostile.sum()))

    return {
        "group": group,
        "n_posts": int(mask.sum()),
        "escalation": escalation,
        "de_escalation": de_escalation,
        "model_hate_share": model_share,
        "human_hate_share": human_share,
    }


def directional_table(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    task: TaskConfig,
    groups: tuple[str, ...] = ANALYSIS_GROUPS,
) -> pd.DataFrame:
    rows = []
    for group in groups:
        m = model_group_metrics(df, y_pred, group, task)
        rows.append({
            "group": group,
            "n_posts": m["n_posts"],
            "n_human_hostile": m["human_hate_share"].n,
            "human_hate_share": m["human_hate_share"].point,
            "model_hate_share": m["model_hate_share"].point,
            "model_share_lo": m["model_hate_share"].lo,
            "model_share_hi": m["model_hate_share"].hi,
            "n_model_hostile": m["model_hate_share"].n,
            "escalation": m["escalation"].point,
            "n_human_offensive": m["escalation"].n,
            "de_escalation": m["de_escalation"].point,
            "n_human_hate": m["de_escalation"].n,
            "reliable": m["human_hate_share"].reliable,
        })
    return pd.DataFrame(rows)


def group_metrics_json(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    task: TaskConfig,
    groups: tuple[str, ...] = ANALYSIS_GROUPS,
) -> dict:
    out: dict[str, dict] = {}
    for group in groups:
        m = model_group_metrics(df, y_pred, group, task)
        out[group] = {
            "n_posts": m["n_posts"],
            "escalation_rate": m["escalation"].as_dict(),
            "de_escalation_rate": m["de_escalation"].as_dict(),
            "model_hate_share": m["model_hate_share"].as_dict(),
            "human_hate_share": m["human_hate_share"].as_dict(),
        }
    return out


# confound: is the group's content simply more contested?


def annotator_agreement(df: pd.DataFrame, group: str) -> dict:
    sub = df[group_mask(df, group)]
    hostile = sub[sub["majority_label"].isin(HOSTILE_LABELS)]
    if hostile.empty:
        return {"n": 0, "unanimity": None, "pairwise_agreement": None,
                "hate_offensive_split": None}

    unanimous = 0
    pairwise: list[float] = []
    split = 0
    for labels in hostile["annotator_labels"]:
        labels = list(labels)
        if len(set(labels)) == 1:
            unanimous += 1
        pairs = [
            (labels[i], labels[j])
            for i in range(len(labels))
            for j in range(i + 1, len(labels))
        ]
        if pairs:
            pairwise.append(sum(a == b for a, b in pairs) / len(pairs))
        if "hatespeech" in labels and "offensive" in labels:
            split += 1

    n = len(hostile)
    return {
        "n": n,
        "unanimity": unanimous / n,
        "pairwise_agreement": float(np.mean(pairwise)) if pairwise else None,
        "hate_offensive_split": split / n,
        "unanimity_ci": wilson(unanimous, n).as_dict(),
        "split_ci": wilson(split, n).as_dict(),
    }


def agreement_table(
    df: pd.DataFrame, groups: tuple[str, ...] = ANALYSIS_GROUPS
) -> pd.DataFrame:
    rows = []
    for group in groups:
        a = annotator_agreement(df, group)
        rows.append({
            "group": group,
            "n_hostile": a["n"],
            "unanimity": a["unanimity"],
            "pairwise_agreement": a["pairwise_agreement"],
            "hate_offensive_split": a["hate_offensive_split"],
        })
    return pd.DataFrame(rows)


# gradient agreement


def gradient_agreement(table: pd.DataFrame, min_n: int = MIN_GROUP_N) -> dict:
    import itertools

    from scipy import stats

    usable = table[
        (table["n_human_hostile"] >= min_n)
        & table["human_hate_share"].notna()
        & table["model_hate_share"].notna()
    ]
    if len(usable) < 3:
        return {"n_groups": len(usable), "spearman_rho": None, "p_value": None}

    human = usable["human_hate_share"].to_numpy()
    model = usable["model_hate_share"].to_numpy()
    rho = float(stats.spearmanr(human, model).statistic)

    # Exact permutation p for small n
    n = len(usable)
    if n <= 8:
        perms = [
            abs(stats.spearmanr(human, list(perm)).statistic)
            for perm in itertools.permutations(model)
        ]
        p = float(sum(v >= abs(rho) - 1e-12 for v in perms) / len(perms))
        p_method = f"exact permutation ({len(perms)} orderings)"
    else:
        p = float(stats.spearmanr(human, model).pvalue)
        p_method = "t-approximation"

    shift = usable["model_hate_share"] - usable["human_hate_share"]
    return {
        "n_groups": int(len(usable)),
        "groups": usable["group"].tolist(),
        "spearman_rho": rho,
        "p_value": p,
        "p_method": p_method,
        "mean_shift": float(shift.mean()),
        "max_positive_shift": {
            "group": usable.loc[shift.idxmax(), "group"],
            "shift": float(shift.max()),
        },
        "max_negative_shift": {
            "group": usable.loc[shift.idxmin(), "group"],
            "shift": float(shift.min()),
        },
    }


__all__ = [
    "Proportion",
    "agreement_table",
    "annotator_agreement",
    "available_groups",
    "directional_table",
    "gradient_agreement",
    "group_mask",
    "group_metrics_json",
    "human_baseline_table",
    "human_hate_share",
    "model_group_metrics",
    "wilson",
]
