from __future__ import annotations

import argparse
import sys

from src import config, data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--rebuild-splits",
        action="store_true",
        help="overwrite data/processed/splits.json if it already exists",
    )
    args = parser.parse_args()

    paths = data.download_all(force=args.force_download)
    for name, path in paths.items():
        print(f"  {name:12s} {path}  ({path.stat().st_size / 1e6:.1f} MB)")

    # sanity check against the pilot
    print("\nHateXplain")
    hx = data.load_hatexplain()
    hx_summary = data.hatexplain_summary(hx)

    print("HateCheck (Muslim-targeted subset -- the one large enough to use)")
    hc = data.hatecheck_muslim()
    hc_summary = data.hatecheck_summary(hc)

    observed = {**hx_summary, **hc_summary}
    ok = True
    print(f"\n{'metric':32s} {'expected':>10s} {'observed':>10s}   status")
    print("-" * 68)
    for key, expected in config.EXPECTED.items():
        got = observed[key]
        match = got == expected
        ok &= match
        print(f"{key:32s} {expected:10d} {got:10d}   {'ok' if match else 'MISMATCH'}")

    if not ok:
        print("\nData loader does not reproduce the pilot counts -- stopping.")
        return 1
    print("\nAll pilot counts reproduced.")

    # Splits 
    
    split_path = config.PROCESSED_DIR / config.DEFAULT_SPLIT.filename
    if split_path.exists() and not args.rebuild_splits:
        print(f"\nSplits already exist at {split_path}")
        print("  Not rebuilding: the archived binary results were scored on "
              "these exact rows.")
        print("  Use --rebuild-splits only if you intend to invalidate them.")
        splits = data.load_splits(split_path)
    else:
        stratified_on = data.labelled_hatexplain(task=config.SPLIT_TASK)
        splits = data.make_splits(stratified_on, config.DEFAULT_SPLIT)
        data.save_splits(splits, split_path)
        print(f"\nSplits written to {split_path}")

    labelled = data.labelled_hatexplain()
    train, val, test = data.apply_splits(labelled, splits)

    task = config.DEFAULT_TASK
    print(f"\nTask: {task.name} -- {task.description}")
    print(f"{'split':8s} {'n':>7s} " + " ".join(
        f"{name:>11s}" for name in task.class_names
    ) + f" {'islam':>7s}")
    print("-" * (16 + 12 * task.n_classes + 8))
    for name, part in (("train", train), ("val", val), ("test", test)):
        counts = part["label"].value_counts()
        cells = " ".join(
            f"{int(counts.get(i, 0)):11d}" for i in range(task.n_classes)
        )
        print(f"{name:8s} {len(part):7d} {cells} "
              f"{int(part['is_islam_slice'].sum()):7d}")

    print(f"\nseed = {config.SEED}, split file = {split_path.name}")
    print(f"class mapping: {task.label_map}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
