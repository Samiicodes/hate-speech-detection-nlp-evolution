from __future__ import annotations

import argparse
import sys

from src import config, data
from src.harness import EvalInputs, evaluate, print_report
from src.models.naive_bayes import NaiveBayesDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bigrams", action="store_true",
                        help="ablation: word 1-2 grams instead of unigrams")
    parser.add_argument("--empirical-prior", action="store_true",
                        help="sensitivity check: fit the class prior from the "
                             "data instead of holding it uniform")
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    name = args.name or "naive_bayes"
    if args.bigrams:
        name += "_bigram"
    if args.empirical_prior:
        name += "_empirical_prior"

    train, val, test = data.get_dataset()
    hatecheck = data.hatecheck_muslim()

    detector = NaiveBayesDetector(
        name=name,
        ngram_range=(1, 2) if args.bigrams else (1, 1),
        fit_prior=args.empirical_prior,
        tune_grid=None if args.no_tune else
        (0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
    )

    report = evaluate(
        detector,
        EvalInputs(train=train, val=val, test=test, hatecheck=hatecheck),
        notes=(
            "Multinomial Naive Bayes on raw term counts. Uniform class prior "
            "(the NB analogue of the balanced class weights used by the "
            "discriminative systems). Smoothing selected on the validation "
            "split; test never used."
        ),
    )
    print_report(report)

    if detector.tuning_history:
        print("\nValidation sweep over smoothing alpha:")
        for row in detector.tuning_history:
            marker = " <-- chosen" if row["alpha"] == detector.alpha else ""
            print(f"  alpha={row['alpha']:<6} macro-F1 "
                  f"{row['val_macro_f1']:.4f}{marker}")

    top = detector.top_features(k=12, class_names=config.DEFAULT_TASK.class_names)
    print("\nMost distinctive tokens per class:")
    for direction, feats in top.items():
        print(f"  {direction:22s} " + ", ".join(f for f, _ in feats))

    print(f"\nResults written to {config.results_path(name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
