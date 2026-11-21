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
from scripts.run_mhs_transfer import (
    MAPPINGS, gold_labels, hate_share, mhs_human_ordering,
)

NAME = "tfidf_logistic_regression"
PEERS = ("tfidf_linear_svc", "rule_based_lexicon", "naive_bayes",
         "roberta_base", "llm_zero_shot_haiku_corpus")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    from scipy.stats import spearmanr
    from sklearn.metrics import accuracy_score, f1_score
    from src.models.classical_ml import TfidfLinearDetector

    task = config.DEFAULT_TASK
    O, N = task.index_of("offensive"), task.index_of("normal")
    mhs = load()
    text = mhs.drop_duplicates("comment_id").set_index("comment_id")["text"].sort_index()
    targets = comment_targets(mhs, MAPPING).reindex(text.index)
    train, val, _ = data.get_dataset()

    det = TfidfLinearDetector(name=NAME, classifier="logistic_regression")
    t0 = time.perf_counter()
    det.fit(train["text"].tolist(), train["label"].tolist())
    det.tune(val["text"].tolist(), val["label"].tolist())
    preds = np.asarray(det.predict(text.tolist()), dtype=int)
    print(f"{NAME}: fit+tune+infer {time.perf_counter() - t0:.0f}s   C={det.C}")
    pd.DataFrame({"comment_id": text.index, "pred": preds}).to_parquet(
        config.RESULTS_DIR / "mhs_logreg_predictions.parquet", index=False)

    hx_human = pd.Series(HATEXPLAIN_ORDER)
    order = hx_human.sort_values(ascending=False).index
    mhs_human = mhs_human_ordering(mhs, targets)
    hx_model = pd.read_csv(
        config.RESULTS_DIR / "group_hate_share.csv").set_index("group")

    share = pd.Series({c: hate_share(preds[targets[c].fillna(False).to_numpy()])
                       for c in targets.columns})
    print("\nHATE-SHARE PER COMMUNITY (HateXplain order)")
    prev = pd.read_csv(config.RESULTS_DIR / "mhs_transfer_hate_share.csv",
                       index_col=0)
    cmp = pd.DataFrame({NAME: share[order],
                        "tfidf_linear_svc": prev.loc["tfidf_linear_svc", order]})
    cmp["difference"] = cmp[NAME] - cmp["tfidf_linear_svc"]
    print(cmp.round(3).to_string())

    print("\nTHREE CORRELATIONS")
    a = spearmanr(share[order], hx_human[order])
    b = spearmanr(share[order], mhs_human[order])
    c = spearmanr(share[order], hx_model[NAME][order])
    print(f"  (a) vs HX human   {a[0]:+.3f} (p{a[1]:.3f})")
    print(f"  (b) vs MHS human  {b[0]:+.3f} (p{b[1]:.3f})")
    print(f"  (c) vs own HX     {c[0]:+.3f} (p{c[1]:.3f})")
    print(f"  agreement with the SVC's MHS ordering: "
          f"{spearmanr(share[order], prev.loc['tfidf_linear_svc', order])[0]:+.3f}")

    print("\nPERFORMANCE, ALL THREE MAPPINGS")
    perf_prev = pd.read_csv(config.RESULTS_DIR / "mhs_transfer_performance.csv")
    rows = []
    for scheme in MAPPINGS:
        gold = gold_labels(mhs, scheme).reindex(text.index).to_numpy()
        labels = [0, 1, 2] if scheme != "binary" else [0, 2]
        p = np.where(preds == O, N, preds) if scheme == "binary" else preds
        acc = accuracy_score(gold, p)
        mf1 = f1_score(gold, p, average="macro", labels=labels, zero_division=0)
        per = f1_score(gold, p, average=None, labels=labels, zero_division=0)
        rows.append({"mapping": scheme, "system": NAME, "accuracy": float(acc),
                     "macro_f1": float(mf1), "f1_hate": float(per[0]),
                     "f1_off": float(per[1]) if scheme != "binary" else None,
                     "f1_norm": float(per[-1])})
        sub = perf_prev[perf_prev["mapping"] == scheme].set_index("system")
        print(f"\n--- {scheme} ---")
        print(f"  {NAME:30s} acc {acc:.4f}  macro-F1 {mf1:.4f}")
        for peer in PEERS:
            if peer in sub.index:
                r = sub.loc[peer]
                print(f"  {peer:30s} acc {r['accuracy']:.4f}  "
                      f"macro-F1 {r['macro_f1']:.4f}")

    out = config.RESULTS_DIR / "mhs_logreg_performance.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    share.to_csv(config.RESULTS_DIR / "mhs_logreg_hate_share.csv")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
