from __future__ import annotations

import argparse
import sys
import time

from src import config, crossval, data

MODELS = (
    "rule_based_lexicon",
    "naive_bayes",
    "tfidf_linear_svc",
    "tfidf_logistic_regression",
    "roberta_base",
)

# deterministic, one pass
SINGLE_SEED_MODELS = {
    "rule_based_lexicon", "naive_bayes", "tfidf_linear_svc",
    "tfidf_logistic_regression",
}


def make_builder(model: str, args) -> callable:
    if model == "rule_based_lexicon":
        from src.models.rule_based import LexiconDetector

        return lambda fold, seed: LexiconDetector(name=f"{model}_fold{fold}")

    if model == "naive_bayes":
        from src.models.naive_bayes import NaiveBayesDetector

        return lambda fold, seed: NaiveBayesDetector(name=f"{model}_fold{fold}")

    if model in ("tfidf_linear_svc", "tfidf_logistic_regression"):
        from src.models.classical_ml import TfidfLinearDetector

        classifier = (
            "linear_svc" if model.endswith("linear_svc") else "logistic_regression"
        )
        return lambda fold, seed: TfidfLinearDetector(
            name=f"{model}_fold{fold}", classifier=classifier, seed=seed,
        )

    if model == "roberta_base":
        from src.models.transformer import TransformerDetector, require_torch

        require_torch()
        return lambda fold, seed: TransformerDetector(
            name=f"{model}_fold{fold}_seed{seed}",
            model_name=args.hf_model,
            seed=seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )

    raise ValueError(f"unknown model {model!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default=None)
    parser.add_argument("--all-fast", action="store_true",
                        help="run every model except roberta_base")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--build-folds", action="store_true")
    parser.add_argument("--rebuild-folds", action="store_true")
    parser.add_argument("--hf-model", default="roberta-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    corpus = data.labelled_hatexplain(task=config.DEFAULT_TASK)

    # folds
    if args.build_folds or args.rebuild_folds or not crossval.FOLDS_FILE.exists():
        if crossval.FOLDS_FILE.exists() and not args.rebuild_folds:
            folds = crossval.load_folds()
            print(f"Folds already exist at {crossval.FOLDS_FILE}")
        else:
            folds = crossval.make_folds(corpus, n_splits=args.n_splits)
            crossval.save_folds(folds)
            print(f"Built {folds.n_splits} folds over {len(corpus)} posts "
                  f"-> {crossval.FOLDS_FILE}")
    else:
        folds = crossval.load_folds()

    counts = {}
    for f in folds.fold:
        counts[f] = counts.get(f, 0) + 1
    print(f"fold sizes: {[counts[k] for k in sorted(counts)]}  "
          f"(seed {folds.seed}, stratified on {folds.stratify})")

    if args.build_folds and not args.model and not args.all_fast:
        return 0

    targets = list(MODELS[:-1]) if args.all_fast else ([args.model] if args.model
                                                       else [])
    if not targets:
        print("Nothing to run. Pass --model or --all-fast.")
        return 1

    corpus = corpus.sort_values("post_id", kind="mergesort").reset_index(drop=True)

    for model in targets:
        seeds = [args.seeds[0]] if model in SINGLE_SEED_MODELS else args.seeds
        build = make_builder(model, args)
        print(f"\n{'#' * 68}\n# {model}: {len(seeds)} seed(s) x "
              f"{folds.n_splits} folds\n{'#' * 68}")

        predictions: dict[int, list[int]] = {}
        started = time.perf_counter()
        for seed in seeds:
            print(f"\nseed {seed}")
            preds = crossval.out_of_fold_predictions(
                corpus, folds, build, seed=seed
            )
            predictions[seed] = preds.tolist()
            agree = (preds == corpus["label"].to_numpy()).mean()
            print(f"  out-of-fold accuracy (diagnostic only): {agree:.4f}")

        elapsed = time.perf_counter() - started
        path = crossval.save_oof(
            model, corpus, predictions, folds,
            notes=(
                f"{folds.n_splits}-fold out-of-fold predictions over the full "
                f"corpus, {len(seeds)} seed(s), {elapsed:.0f}s. For the "
                "target-group analysis only; headline numbers come from the "
                "held-out split."
            ),
        )
        print(f"\nWritten: {path}  ({elapsed / 60:.1f} min)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
