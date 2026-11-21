from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src import config

REPO = "ucberkeley-dlab/measuring-hate-speech"
PARQUET = "measuring-hate-speech.parquet"

# Community to MHS subgroups
MAPPING: dict[str, tuple[str, ...]] = {
    "jewish":     ("target_religion_jewish",),
    "african":    ("target_race_black",),
    "hispanic":   ("target_race_latinx",),
    "arab":       ("target_race_middle_eastern",),
    "islam":      ("target_religion_muslim",),
    "asian":      ("target_race_asian",),
    # HateXplain's category is "Homosexual", which covers gay and lesbian.
    "homosexual": ("target_sexuality_gay", "target_sexuality_lesbian"),
    # HateXplain's category is literally "Refugee/Immigrant".
    "refugee":    ("target_origin_immigrant", "target_origin_undocumented",
                   "target_origin_migrant_worker"),
    "women":      ("target_gender_women",),
    "caucasian":  ("target_race_white",),
}

# sensitivity readings only
SENSITIVITY: dict[str, tuple[str, ...]] = {
    "homosexual": ("target_sexuality_gay",),
    "refugee":    ("target_origin_immigrant",),
}

# The published HateXplain human hate-share, for the ordering comparison.
HATEXPLAIN_ORDER = {
    "jewish": 0.772, "african": 0.758, "hispanic": 0.754, "arab": 0.722,
    "islam": 0.660, "asian": 0.547, "homosexual": 0.466, "refugee": 0.328,
    "women": 0.299, "caucasian": 0.286,
}


def load() -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    return pd.read_parquet(hf_hub_download(REPO, PARQUET, repo_type="dataset"))


def comment_targets(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    out = {}
    for community, cols in mapping.items():
        present = [c for c in cols if c in df.columns]
        flag = df[present].fillna(0).astype(bool).any(axis=1)
        out[community] = flag.groupby(df["comment_id"]).any()
    return pd.DataFrame(out)


def measures(df: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    ann = df.merge(targets, left_on="comment_id", right_index=True, how="left")
    per_comment = df.groupby("comment_id")["hate_speech_score"].mean()

    rows = []
    for community in targets.columns:
        mask = ann[community].fillna(False).to_numpy()
        sub = ann[mask]
        hostile = sub["hatespeech"].isin([1.0, 2.0])
        n_hostile = int(hostile.sum())
        cids = targets.index[targets[community].to_numpy()]
        scores = per_comment.reindex(cids).dropna()
        rows.append({
            "community": community,
            "n_comments": int(len(cids)),
            "n_annotations": int(len(sub)),
            "A_hate_share": float((sub.loc[hostile, "hatespeech"] == 2.0).mean())
            if n_hostile else np.nan,
            "n_hostile_annotations": n_hostile,
            "C_mean_score": float(scores.mean()) if len(scores) else np.nan,
            "C_sd": float(scores.std()) if len(scores) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-n", type=int, default=200,
                    help="reliability floor on comments per community")
    args = ap.parse_args()

    df = load()
    print(f"MHS: {len(df):,} annotations | {df.comment_id.nunique():,} comments "
          f"| {df.annotator_id.nunique():,} annotators")

    targets = comment_targets(df, MAPPING)
    tab = measures(df, targets).sort_values("C_mean_score", ascending=False)
    tab["hatexplain_share"] = tab["community"].map(HATEXPLAIN_ORDER)
    tab["below_floor"] = tab["n_comments"] < args.min_n

    print("\n" + "=" * 88)
    print("GROUP MEASURES  (sorted by MHS mean score; no models involved)")
    print("=" * 88)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    keep = tab[~tab["below_floor"]].dropna(subset=["A_hate_share", "C_mean_score"])
    print(f"\nranking over {len(keep)} communities clearing n >= {args.min_n}")
    _spearman(keep, "A_hate_share")
    _spearman(keep, "C_mean_score")

    print("\n" + "=" * 88)
    print("SENSITIVITY -- narrower reading of the two judgement calls")
    print("=" * 88)
    alt = dict(MAPPING) | SENSITIVITY
    alt_tab = measures(df, comment_targets(df, alt))
    both = tab.merge(alt_tab, on="community", suffixes=("", "_alt"))
    print(both[["community", "n_comments", "n_comments_alt",
                "A_hate_share", "A_hate_share_alt",
                "C_mean_score", "C_mean_score_alt"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out = config.RESULTS_DIR / "mhs_group_severity.csv"
    tab.to_csv(out, index=False)
    print(f"\nwritten to {out}")
    return 0


def _spearman(keep: pd.DataFrame, col: str) -> None:
    from scipy.stats import spearmanr

    rho, p = spearmanr(keep[col], keep["hatexplain_share"])
    print(f"  Spearman({col:14s}, HateXplain hate-share) = {rho:+.3f}  p = {p:.4f}")


if __name__ == "__main__":
    sys.exit(main())
