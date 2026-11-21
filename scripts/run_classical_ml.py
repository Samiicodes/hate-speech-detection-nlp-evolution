from __future__ import annotations

import argparse
import sys

from src import config, data
from src.harness import EvalInputs, evaluate, print_report
from src.models.classical_ml import TfidfLinearDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier", default="linear_svc",
        choices=["linear_svc", "logistic_regression"],
    )
    parser.add_argument("--no-char", action="store_true",
                        help="word n-grams only, no character n-grams")
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or f"tfidf_{args.classifier}"
    if args.no_char:
        name += "_word_only"

    train, val, test = data.get_dataset()
    hatecheck = data.hatecheck_muslim()

    detector = TfidfLinearDetector(
        name=name,
        classifier=args.classifier,
        use_char=not args.no_char,
        tune_grid=None if args.no_tune else (0.01, 0.1, 0.5, 1.0, 2.0, 5.0),
    )

    report = evaluate(
        detector,
        EvalInputs(train=train, val=val, test=test, hatecheck=hatecheck),
        notes="C selected on the validation split by macro-F1; test never used.",
    )
    print_report(report)

    if detector.tuning_history:
        print("\nValidation sweep over C:")
        for row in detector.tuning_history:
            marker = " <-- chosen" if row["C"] == detector.C else ""
            print(f"  C={row['C']:<6} macro-F1 {row['val_macro_f1']:.4f}{marker}")

    top = detector.top_features(k=15, class_names=config.DEFAULT_TASK.class_names)
    print("\nStrongest features per class:")
    for direction, feats in top.items():
        print(f"  {direction:24s} " + ", ".join(f for f, _ in feats))

    print(f"\nResults written to {config.results_path(name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
