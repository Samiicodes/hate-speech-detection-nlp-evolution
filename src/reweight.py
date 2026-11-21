from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ANALYSIS_GROUPS, HOSTILE_LABELS
from .groups import group_mask


def corpus_hate_share(df: pd.DataFrame) -> float:
    hostile = df[df["majority_label"].isin(HOSTILE_LABELS)]
    return float((hostile["majority_label"] == "hatespeech").mean())


def group_balance_weights(
    df: pd.DataFrame,
    groups: tuple[str, ...] = ANALYSIS_GROUPS,
    target_share: float | None = None,
    normalise: bool = True,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> tuple[np.ndarray, dict]:
    target = corpus_hate_share(df) if target_share is None else target_share

    is_hate = (df["majority_label"] == "hatespeech").to_numpy()
    is_off = (df["majority_label"] == "offensive").to_numpy()

    masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    per_group: dict[str, dict] = {}
    for group in groups:
        mask = group_mask(df, group)
        g_hate, g_off = mask & is_hate, mask & is_off
        n_h, n_o = int(g_hate.sum()), int(g_off.sum())
        per_group[group] = {
            "n_hate": n_h,
            "n_offensive": n_o,
            "share_before": (n_h / (n_h + n_o)) if (n_h + n_o) else None,
        }
        if n_h and n_o:
            masks[group] = (g_hate, g_off)
        else:
            per_group[group]["note"] = "degenerate -- one class empty, skipped"

    # iterative proportional fitting
    weights = np.ones(len(df), dtype=float)
    deviation = float("inf")
    iterations = 0
    for iterations in range(1, max_iter + 1):
        for g_hate, g_off in masks.values():
            wh, wo = weights[g_hate].sum(), weights[g_off].sum()
            if wh <= 0 or wo <= 0:
                continue
            total = wh + wo
            weights[g_hate] *= target * total / wh
            weights[g_off] *= (1.0 - target) * total / wo
        deviation = max(
            abs(weights[g_hate].sum() / (weights[g_hate].sum() + weights[g_off].sum())
                - target)
            for g_hate, g_off in masks.values()
        )
        if deviation <= tol:
            break

    touched = np.zeros(len(df), dtype=bool)
    for g_hate, g_off in masks.values():
        touched |= g_hate | g_off

    if normalise:
        weights = weights / weights.mean()

    for group, (g_hate, g_off) in masks.items():
        wh, wo = weights[g_hate].sum(), weights[g_off].sum()
        per_group[group]["share_after"] = float(wh / (wh + wo))
        per_group[group]["mean_w_hate"] = float(weights[g_hate].mean())
        per_group[group]["mean_w_offensive"] = float(weights[g_off].mean())

    # Kish effective sample size
    ess = float(weights.sum() ** 2 / np.square(weights).sum())

    report = {
        "method": "instance reweighting by iterative proportional fitting",
        "target_hate_share": target,
        "n_posts": int(len(df)),
        "n_reweighted": int(touched.sum()),
        "n_untouched": int((~touched).sum()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "effective_sample_size": ess,
        "ess_fraction": ess / len(df),
        "ipf_iterations": iterations,
        "ipf_max_deviation": float(deviation),
        "ipf_converged": bool(deviation <= tol),
        "multi_target_rule": (
            "groups overlap, so targets are fitted jointly by IPF rather than "
            "applied independently"
        ),
        "normal_posts": "weight 1 -- correction targets hate/offensive only",
        "per_group": per_group,
    }
    return weights, report


def verify_correction(
    df: pd.DataFrame,
    weights: np.ndarray,
    groups: tuple[str, ...] = ANALYSIS_GROUPS,
) -> pd.DataFrame:
    rows = []
    is_hate = (df["majority_label"] == "hatespeech").to_numpy()
    is_hostile = df["majority_label"].isin(HOSTILE_LABELS).to_numpy()
    for group in groups:
        mask = group_mask(df, group) & is_hostile
        w = weights[mask]
        h = is_hate[mask]
        rows.append({
            "group": group,
            "n_hostile": int(mask.sum()),
            "share_before": float(h.mean()) if mask.sum() else None,
            "share_after_weighted": (
                float(w[h].sum() / w.sum()) if w.sum() else None
            ),
        })
    return pd.DataFrame(rows)


__all__ = ["corpus_hate_share", "group_balance_weights", "verify_correction"]
