from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src import config
from scripts.mhs_feasibility import (
    HATEXPLAIN_ORDER, MAPPING, comment_targets, load,
)

# All load positively
ORDINALS = ("sentiment", "respect", "insult", "humiliate", "status",
            "dehumanize", "violence", "genocide", "attack_defend", "hatespeech")

# Severity ladder
SEVERITY_LADDER = ("humiliate", "status", "dehumanize", "violence", "genocide")

# authors' own band boundaries
IRT_LOW, IRT_HIGH = -1.0, 0.5


def composite(df: pd.DataFrame, cols) -> pd.Series:
    scaled = [df[c] / df[c].max() for c in cols]
    return pd.concat(scaled, axis=1).mean(axis=1)


def hate_share(scores: pd.Series, lo: float, hi: float) -> tuple[float, int]:
    hostile = scores > lo
    if not hostile.any():
        return np.nan, 0
    return float((scores > hi).sum() / hostile.sum()), int(hostile.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    df = load()
    targets = comment_targets(df, MAPPING)

    # comment-level scores on each scale
    irt = df.groupby("comment_id")["hate_speech_score"].first()
    raw_all = composite(df, ORDINALS).groupby(df["comment_id"]).mean()
    raw_sev = composite(df, SEVERITY_LADDER).groupby(df["comment_id"]).mean()

    # percentile-matched cut points
    p_lo = float((irt <= IRT_LOW).mean())
    p_hi = float((irt <= IRT_HIGH).mean())
    print(f"IRT band boundaries {IRT_LOW} / {IRT_HIGH} sit at percentiles "
          f"{100 * p_lo:.1f} / {100 * p_hi:.1f}")
    cuts = {
        "MHS IRT":       (irt, IRT_LOW, IRT_HIGH),
        "MHS raw (10)":  (raw_all, raw_all.quantile(p_lo), raw_all.quantile(p_hi)),
        "MHS raw (sev)": (raw_sev, raw_sev.quantile(p_lo), raw_sev.quantile(p_hi)),
    }
    for name, (s, lo, hi) in cuts.items():
        print(f"  {name:14s} cuts at {lo:+.3f} / {hi:+.3f}")

    # raw hatespeech ordinal, annotation level
    ann = df.merge(targets, left_on="comment_id", right_index=True, how="left")

    rows = []
    for community in targets.columns:
        cids = targets.index[targets[community].to_numpy()]
        row = {"community": community, "n_comments": int(len(cids)),
               "hatexplain": HATEXPLAIN_ORDER[community]}
        for name, (s, lo, hi) in cuts.items():
            share, n_h = hate_share(s.reindex(cids).dropna(), lo, hi)
            row[name] = share
        sub = ann[ann[community].fillna(False).to_numpy()]
        hostile = sub["hatespeech"].isin([1.0, 2.0])
        row["raw hatespeech 0-2"] = (
            float((sub.loc[hostile, "hatespeech"] == 2.0).mean())
            if hostile.any() else np.nan
        )
        rows.append(row)

    tab = pd.DataFrame(rows).sort_values("hatexplain", ascending=False)
    cols = ["community", "n_comments", "hatexplain",
            "MHS raw (10)", "MHS raw (sev)", "raw hatespeech 0-2", "MHS IRT"]

    print("\n" + "=" * 92)
    print("THREE ORDERINGS, SAME GROUPS AND CONDITIONING (sorted by HateXplain)")
    print("=" * 92)
    print(tab[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 92)
    print("PAIRWISE SPEARMAN")
    print("=" * 92)
    from scipy.stats import spearmanr

    measures = ["hatexplain", "MHS raw (10)", "MHS raw (sev)",
                "raw hatespeech 0-2", "MHS IRT"]
    print(f"{'':22s}" + "".join(f"{m:>20s}" for m in measures))
    for a in measures:
        line = f"{a:22s}"
        for b in measures:
            if a == b:
                line += f"{'--':>20s}"
            else:
                rho, p = spearmanr(tab[a], tab[b])
                line += f"{f'{rho:+.3f} (p{p:.2f})':>20s}"
        print(line)

    out = config.RESULTS_DIR / "mhs_irt_check.csv"
    tab[cols].to_csv(out, index=False)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
