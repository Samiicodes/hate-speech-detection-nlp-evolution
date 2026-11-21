from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np

from src import config, crossval, data
from src.groups import group_mask, human_hate_share, wilson

HOSTILE = config.HOSTILE_LABELS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="hispanic")
    parser.add_argument("--compare", nargs="*", default=["islam", "african"])
    args = parser.parse_args()

    task = config.DEFAULT_TASK
    corpus = data.labelled_hatexplain(task=task).sort_values(
        "post_id", kind="mergesort"
    ).reset_index(drop=True)
    g = args.group

    hate_idx = task.index_of("hatespeech")
    off_idx = task.index_of("offensive")

    mask = group_mask(corpus, g)
    sub = corpus[mask]
    hostile = sub[sub["majority_label"].isin(HOSTILE)]
    print(f"=== {g}: {int(mask.sum())} posts, {len(hostile)} hostile")
    print(f"human hate-share: {human_hate_share(corpus, g)}\n")

    # 1. seed stability + 2
    print("--- model hate-share per seed, with Wilson intervals")
    for model in crossval.available_oof():
        blob = crossval.load_oof(model)
        print(f"  {model}")
        for seed in blob["seeds"]:
            pred = np.asarray(blob["predictions"][str(seed)], dtype=int)[mask]
            k = int((pred == hate_idx).sum())
            n = int(((pred == hate_idx) | (pred == off_idx)).sum())
            print(f"    seed {seed}: {wilson(k, n)}")

    human = human_hate_share(corpus, g)
    print(f"\n  human:        {human}")
    print("  Overlapping intervals mean the shift is not distinguishable.")

    # 3. multi-target contamination
    print(f"\n--- what else are {g}-tagged hostile posts tagged with?")
    co = Counter()
    for groups in hostile["target_groups"]:
        for other in groups:
            if other != g:
                co[other] += 1
    n_h = len(hostile)
    print(f"  {n_h} hostile posts; {sum(1 for gs in hostile['target_groups'] if len(gs) == 1)} "
          f"tagged with {g} alone")
    for other, count in co.most_common(8):
        print(f"    {other:14s} {count:5d}  ({100 * count / n_h:5.1f}% of {g} hostile)")

    # 4. exclusive subset
    print(f"\n--- restricted to posts tagged {g} and nothing else")
    solo = hostile[hostile["target_groups"].map(lambda gs: gs == (g,))]
    if solo.empty:
        print("  none")
        return 0
    solo_idx = corpus.index.isin(solo.index)
    k_h = int((solo["majority_label"] == "hatespeech").sum())
    print(f"  human: {wilson(k_h, len(solo))}")
    for model in crossval.available_oof():
        blob = crossval.load_oof(model)
        seed = blob["seeds"][0]
        pred = np.asarray(blob["predictions"][str(seed)], dtype=int)[solo_idx]
        k = int((pred == hate_idx).sum())
        n = int(((pred == hate_idx) | (pred == off_idx)).sum())
        print(f"  {model:28s} {wilson(k, n)}")

    # comparison groups
    if args.compare:
        print("\n--- co-tag overlap with the comparison groups")
        for other in args.compare:
            both = mask & group_mask(corpus, other)
            print(f"  {g} & {other}: {int(both.sum())} posts "
                  f"({100 * both.sum() / max(1, mask.sum()):.1f}% of {g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
