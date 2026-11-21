from __future__ import annotations

import argparse
import json
import sys

from src import config, data
from src.harness import (
    EvalInputs,
    aggregate_reports,
    evaluate,
    print_aggregate,
    print_report,
)
from src.models.transformer import TransformerDetector, require_torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44],
        help="one training run per seed; results reported as mean +/- std",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default=None, help="cuda / cpu / mps")
    parser.add_argument("--no-amp", action="store_true",
                        help="disable mixed precision")
    parser.add_argument("--deterministic", action="store_true",
                        help="exact reproducibility, ~20%% slower")
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--name", default=None)
    parser.add_argument("--limit-train", type=int, default=None,
                        help="subsample the training split (smoke tests only)")
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="re-aggregate the per-seed reports already in results/runs/ "
             "instead of retraining; use after fixing a reporting bug",
    )
    args = parser.parse_args()

    name = args.name or args.model.replace("-", "_").replace("/", "_")

    if args.aggregate_only:
        reports = []
        for seed in args.seeds:
            path = config.results_path(f"{name}_seed{seed}", runs=True)
            if not path.exists():
                print(f"missing {path}")
                return 1
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        aggregate = aggregate_reports(
            reports, name,
            notes=(
                f"Mean over {len(reports)} fine-tuning runs (seeds "
                f"{args.seeds}). Stored predictions are from seed "
                f"{args.seeds[0]}. Re-aggregated from saved per-seed reports."
            ),
        )
        print_aggregate(aggregate)
        print(f"\nAggregate written to {config.results_path(name)}")
        return 0

    require_torch()
    train, val, test = data.get_dataset()
    hatecheck = data.hatecheck_muslim()

    if args.limit_train:
        train = train.head(args.limit_train)
        print(f"WARNING: training on {len(train)} rows only (smoke test)")

    inputs = EvalInputs(train=train, val=val, test=test, hatecheck=hatecheck)
    val_data = (val["text"].tolist(), val["label"].tolist())

    reports = []
    for i, seed in enumerate(args.seeds, start=1):
        run_name = f"{name}_seed{seed}"
        print(f"\n{'#' * 68}")
        print(f"# run {i}/{len(args.seeds)}: {run_name}")
        print(f"{'#' * 68}")

        detector = TransformerDetector(
            name=run_name,
            model_name=args.model,
            seed=seed,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            use_amp=not args.no_amp,
            deterministic=args.deterministic,
            class_weighted=not args.no_class_weight,
            val_data=val_data,
        )

        report = evaluate(
            detector,
            inputs,
            seed=seed,
            notes=(
                f"{args.model}, {args.epochs} epochs, lr {args.lr}, "
                "class-weighted loss. Validation used for per-epoch "
                "monitoring and best-epoch selection only."
            ),
            to_runs_dir=True,
        )
        print_report(report)

        hist = detector.epoch_history
        if hist and "val_macro_f1" in hist[0]:
            print("\n  epoch  train_loss  val_macro_F1")
            for row in hist:
                marker = " <-- kept" if row["epoch"] == detector.best_epoch else ""
                print(
                    f"  {row['epoch']:5d}  {row['train_loss']:10.4f}  "
                    f"{row['val_macro_f1']:12.4f}{marker}"
                )

        reports.append(report)

    if len(reports) == 1:
        # Single seed: copy out of runs
        single = {**reports[0], "model_name": name}
        single.pop("_path", None)
        path = config.results_path(name)
        path.write_text(json.dumps(single, indent=2), encoding="utf-8")
        print(f"\nSingle run written to {path} (no variance estimate)")
        return 0

    aggregate = aggregate_reports(
        reports,
        name,
        notes=(
            f"Mean over {len(reports)} fine-tuning runs (seeds "
            f"{args.seeds}). Stored predictions are from seed {args.seeds[0]}."
        ),
    )
    print_aggregate(aggregate)
    print(f"\nAggregate written to {config.results_path(name)}")
    print(f"Per-seed reports in {config.RUNS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
