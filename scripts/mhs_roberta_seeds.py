from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from src import config, data
from scripts.mhs_feasibility import (
    HATEXPLAIN_ORDER, MAPPING, comment_targets, load,
)
from scripts.run_mhs_transfer import gold_labels, hate_share, mhs_human_ordering


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--mapping", default="bands")
    args = ap.parse_args()

    from scipy.stats import spearmanr
    from sklearn.metrics import f1_score
    from src.models.transformer import TransformerDetector

    mhs = load()
    text = mhs.drop_duplicates("comment_id").set_index("comment_id")["text"].sort_index()
    targets = comment_targets(mhs, MAPPING).reindex(text.index)
    train, val, _ = data.get_dataset()

    hx_human = pd.Series(HATEXPLAIN_ORDER)
    order = hx_human.sort_values(ascending=False).index
    mhs_human = mhs_human_ordering(mhs, targets)
    hx_model = pd.read_csv(
        config.RESULTS_DIR / "group_hate_share.csv").set_index("group")["roberta_base"]
    gold = gold_labels(mhs, args.mapping).reindex(text.index).to_numpy()

    rows, shares = [], {}
    for seed in args.seeds:
        path = config.RESULTS_DIR / f"mhs_roberta_seed{seed}.parquet"
        if path.exists():
            preds = pd.read_parquet(path)["pred"].to_numpy(dtype=int)
            print(f"  seed {seed}: loaded {path.name}")
        else:
            det = TransformerDetector(name=f"roberta_base_seed{seed}", seed=seed)
            t0 = time.perf_counter()
            det.fit(train["text"].tolist(), train["label"].tolist())
            preds = np.asarray(det.predict(text.tolist()), dtype=int)
            pd.DataFrame({"comment_id": text.index, "pred": preds}).to_parquet(
                path, index=False)
            print(f"  seed {seed}: {time.perf_counter() - t0:.0f}s -> {path.name}")

        s = pd.Series({c: hate_share(preds[targets[c].fillna(False).to_numpy()])
                       for c in targets.columns})
        shares[seed] = s
        rows.append({
            "seed": seed,
            "a_vs_hx_human": spearmanr(s[order], hx_human[order])[0],
            "a_p": spearmanr(s[order], hx_human[order])[1],
            "b_vs_mhs_human": spearmanr(s[order], mhs_human[order])[0],
            "c_vs_own_hx": spearmanr(s[order], hx_model[order])[0],
            "macro_f1": f1_score(gold, preds, average="macro", zero_division=0),
        })

    df = pd.DataFrame(rows)
    print("\nPER SEED")
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\nMEAN / SD ACROSS SEEDS")
    for col in ("a_vs_hx_human", "b_vs_mhs_human", "c_vs_own_hx", "macro_f1"):
        m, sd = df[col].mean(), df[col].std(ddof=1)
        print(f"  {col:16s} {m:+.4f}  sd {sd:.4f}   range [{df[col].min():+.4f}, "
              f"{df[col].max():+.4f}]")

    # correlation of the seed-averaged shares -- what run_mhs_transfer reports
    mean_share = pd.DataFrame(shares).mean(axis=1)
    r, p = spearmanr(mean_share[order], hx_human[order])
    print(f"\ncorrelation OF THE MEAN share vs HX human: {r:+.4f} (p {p:.4f})")
    print("  (not the same quantity as the mean of the per-seed correlations)")

    print("\nPER-COMMUNITY HATE-SHARE BY SEED")
    sh = pd.DataFrame(shares).loc[order]
    sh["mean"] = sh.mean(axis=1)
    sh["sd"] = sh[args.seeds].std(axis=1, ddof=1)
    print(sh.round(4).to_string())

    df.to_csv(config.RESULTS_DIR / "mhs_roberta_seed_stability.csv", index=False)
    sh.to_csv(config.RESULTS_DIR / "mhs_roberta_seed_hate_share.csv")
    print("\nwritten to results/mhs_roberta_seed_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
