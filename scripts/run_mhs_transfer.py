from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from src import config, data
from scripts.mhs_feasibility import (
    HATEXPLAIN_ORDER, MAPPING, comment_targets, load,
)
from scripts.run_hatecheck_all import build_detectors, predictions_for

# MHS gold to our three classes
MAPPINGS = ("bands", "ordinal", "binary")


def gold_labels(df: pd.DataFrame, scheme: str) -> pd.Series:
    task = config.DEFAULT_TASK
    H, O, N = (task.index_of("hatespeech"), task.index_of("offensive"),
               task.index_of("normal"))
    if scheme == "bands":
        s = df.groupby("comment_id")["hate_speech_score"].first()
        return pd.Series(np.where(s > 0.5, H, np.where(s > -1.0, O, N)),
                         index=s.index)
    if scheme == "ordinal":
        # Ties break toward the milder class
        m = df.groupby("comment_id")["hatespeech"].mean()
        return pd.Series(np.where(m >= 1.5, H, np.where(m >= 0.5, O, N)),
                         index=m.index)
    if scheme == "binary":
        s = df.groupby("comment_id")["hate_speech_score"].first()
        return pd.Series(np.where(s > 0.5, H, N), index=s.index)
    raise ValueError(scheme)


def hate_share(preds: np.ndarray) -> float:
    task = config.DEFAULT_TASK
    H, O = task.index_of("hatespeech"), task.index_of("offensive")
    hostile = np.isin(preds, (H, O))
    return float((preds[hostile] == H).mean()) if hostile.any() else np.nan


def mhs_human_ordering(df: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    per = df.groupby("comment_id")["hate_speech_score"].first()
    out = {}
    for community in targets.columns:
        s = per.reindex(targets.index[targets[community].to_numpy()]).dropna()
        hostile = s > -1.0
        out[community] = float((s > 0.5).sum() / hostile.sum()) if hostile.any() else np.nan
    return pd.Series(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    args = ap.parse_args()

    task = config.DEFAULT_TASK
    mhs = load()
    targets = comment_targets(mhs, MAPPING)
    text = mhs.drop_duplicates("comment_id").set_index("comment_id")["text"].sort_index()
    targets = targets.reindex(text.index)
    print(f"MHS: {len(text):,} comments, {int(targets.any(axis=1).sum()):,} mapped")

    train, val, _ = data.get_dataset()
    frame = pd.DataFrame({"test_case": text.to_numpy()})

    preds: dict[str, list[np.ndarray]] = {}
    for name, det in build_detectors(set(args.skip), args.seeds):
        t0 = time.perf_counter()
        preds.setdefault(name, []).append(
            predictions_for(det, train, val, frame))
        print(f"  {det.name:24s} {time.perf_counter() - t0:7.1f}s")

    llm_path = config.RESULTS_DIR / "mhs_llm_predictions.parquet"
    if llm_path.exists():
        lp = pd.read_parquet(llm_path).set_index("comment_id")["pred"]
        preds["llm_zero_shot_haiku_corpus"] = [
            lp.reindex(text.index).to_numpy(dtype=int)]
        print(f"  {'llm_zero_shot':24s} loaded from {llm_path.name}")

    # orderings
    hx_human = pd.Series(HATEXPLAIN_ORDER)
    mhs_human = mhs_human_ordering(mhs, targets)
    hx_model = pd.read_csv(config.RESULTS_DIR / "group_hate_share.csv").set_index("group")

    rows, orderings = [], {}
    for name, runs in preds.items():
        per_community = {}
        for community in targets.columns:
            m = targets[community].fillna(False).to_numpy()
            per_community[community] = float(np.mean(
                [hate_share(r[m]) for r in runs]))
        s = pd.Series(per_community)
        orderings[name] = s
        rows.append({"system": name, **{k: v for k, v in s.items()}})

    print("\n" + "=" * 92)
    print("MODEL HATE-SHARE PER COMMUNITY, ON MHS TEXT")
    print("=" * 92)
    tab = pd.DataFrame(rows).set_index("system")
    order = hx_human.sort_values(ascending=False).index
    print(tab[order].round(3).to_string())

    print("\n" + "=" * 92)
    print("THREE CORRELATIONS (Spearman over the ten communities)")
    print("=" * 92)
    from scipy.stats import spearmanr

    print(f"{'system':30s} {'(a) vs HX human':>18s} {'(b) vs MHS human':>18s} "
          f"{'(c) vs own HX':>18s}")
    corr = []
    for name, s in orderings.items():
        a = spearmanr(s[order], hx_human[order])
        b = spearmanr(s[order], mhs_human[order])
        own = hx_model[name] if name in hx_model.columns else None
        c = spearmanr(s[order], own[order]) if own is not None else (np.nan, np.nan)
        corr.append({"system": name, "a_vs_hx_human": a[0], "a_p": a[1],
                     "b_vs_mhs_human": b[0], "b_p": b[1],
                     "c_vs_own_hx": c[0], "c_p": c[1]})
        print(f"{name:30s} {f'{a[0]:+.3f} (p{a[1]:.2f})':>18s} "
              f"{f'{b[0]:+.3f} (p{b[1]:.2f})':>18s} "
              f"{f'{c[0]:+.3f} (p{c[1]:.2f})':>18s}")

    # performance under each mapping
    from sklearn.metrics import accuracy_score, f1_score

    print("\n" + "=" * 92)
    print("PERFORMANCE ON MHS, BY GOLD MAPPING")
    print("=" * 92)
    O = task.index_of("offensive")
    N = task.index_of("normal")
    perf = []
    for scheme in MAPPINGS:
        gold = gold_labels(mhs, scheme).reindex(text.index).to_numpy()
        # Binary gold needs binary predictions
        labels = [0, 1, 2] if scheme != "binary" else [0, 2]
        def fold(r):
            return np.where(r == O, N, r) if scheme == "binary" else r

        print(f"\n--- {scheme} ---   class balance "
              f"{np.round(np.bincount(gold, minlength=3) / len(gold), 3)}"
              f"{'   (macro over 2 classes)' if scheme == 'binary' else ''}")
        head = f"{'system':30s} {'accuracy':>9s} {'macro-F1':>9s} {'F1 hate':>9s}"
        print(head + (f" {'F1 off':>9s} {'F1 norm':>9s}" if scheme != "binary"
                      else f" {'F1 norm':>9s}"))
        for name, runs in preds.items():
            folded = [fold(r) for r in runs]
            accs = [accuracy_score(gold, r) for r in folded]
            f1s = [f1_score(gold, r, average="macro", labels=labels,
                            zero_division=0) for r in folded]
            per = np.mean([f1_score(gold, r, average=None, labels=labels,
                                    zero_division=0) for r in folded], axis=0)
            cells = " ".join(f"{v:9.4f}" for v in per)
            print(f"{name:30s} {np.mean(accs):9.4f} {np.mean(f1s):9.4f} {cells}")
            perf.append({"mapping": scheme, "system": name,
                         "accuracy": float(np.mean(accs)),
                         "macro_f1": float(np.mean(f1s)),
                         "f1_hate": float(per[0]),
                         "f1_off": float(per[1]) if scheme != "binary" else None,
                         "f1_norm": float(per[-1])})

    pd.DataFrame(corr).to_csv(config.RESULTS_DIR / "mhs_transfer_correlations.csv", index=False)
    pd.DataFrame(perf).to_csv(config.RESULTS_DIR / "mhs_transfer_performance.csv", index=False)
    tab.to_csv(config.RESULTS_DIR / "mhs_transfer_hate_share.csv")
    print(f"\nwritten to results/mhs_transfer_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
