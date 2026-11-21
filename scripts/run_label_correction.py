from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src import config, crossval, data
from src.groups import directional_table, gradient_agreement
from src.reweight import group_balance_weights, verify_correction

CORRECTED_SUFFIX = "_corrected"


def _pct(v, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{100 * v:+.1f}" if signed else f"{100 * v:.1f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="tfidf_linear_svc",
        choices=["naive_bayes", "tfidf_linear_svc", "tfidf_logistic_regression",
                 "roberta_base"],
    )
    parser.add_argument("--epochs", type=int, default=3,
                        help="roberta_base only")
    parser.add_argument("--target-share", type=float, default=None,
                        help="default: the corpus-wide hostile hate-share")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task = config.DEFAULT_TASK
    corpus = data.labelled_hatexplain(task=task).sort_values(
        "post_id", kind="mergesort"
    ).reset_index(drop=True)
    folds = crossval.load_folds()
    groups = config.ANALYSIS_GROUPS

    # build the correction
    weights, report = group_balance_weights(
        corpus, groups, target_share=args.target_share
    )
    print("=" * 92)
    print("LABEL CORRECTION BY INSTANCE REWEIGHTING")
    print("=" * 92)
    print(f"  method              {report['method']}")
    print(f"  target hate-share   {_pct(report['target_hate_share'])}%")
    print(f"  posts reweighted    {report['n_reweighted']} of {report['n_posts']}")
    print(f"  weight range        {report['weight_min']:.3f} - "
          f"{report['weight_max']:.3f}")
    print(f"  effective n         {report['effective_sample_size']:.0f} "
          f"({100 * report['ess_fraction']:.1f}% of {report['n_posts']})")
    print(f"  multi-target rule   {report['multi_target_rule']}")

    check = verify_correction(corpus, weights, groups)
    print("\n  per-group hate-share, before and after weighting:")
    print("    " + check.to_string(index=False).replace("\n", "\n    "))
    max_dev = (check["share_after_weighted"] - report["target_hate_share"]).abs().max()
    print(f"\n  max deviation from target after weighting: {max_dev:.6f}")
    if max_dev > 1e-6:
        print("  WARNING: correction did not equalise the shares exactly")

    # out-of-fold, corrected
    if args.model == "roberta_base":
        from src.models.transformer import TransformerDetector, require_torch

        require_torch()

        def build(fold, seed):
            return TransformerDetector(
                name=f"{args.model}{CORRECTED_SUFFIX}_fold{fold}",
                seed=seed, epochs=args.epochs,
            )
    elif args.model == "naive_bayes":
        from src.models.naive_bayes import NaiveBayesDetector

        def build(fold, seed):
            return NaiveBayesDetector(
                name=f"{args.model}{CORRECTED_SUFFIX}_fold{fold}",
            )
    else:
        from src.models.classical_ml import TfidfLinearDetector

        classifier = ("linear_svc" if args.model.endswith("linear_svc")
                      else "logistic_regression")

        def build(fold, seed):
            return TfidfLinearDetector(
                name=f"{args.model}{CORRECTED_SUFFIX}_fold{fold}",
                classifier=classifier, seed=seed,
            )

    print("\n" + "=" * 92)
    print(f"RETRAINING {args.model} ON THE CORRECTED CORPUS")
    print("=" * 92)
    started = time.perf_counter()
    corrected = crossval.out_of_fold_predictions(
        corpus, folds, build, seed=args.seed, sample_weight=weights
    )
    elapsed = time.perf_counter() - started
    print(f"  {elapsed / 60:.1f} min")

    name = f"{args.model}{CORRECTED_SUFFIX}"
    crossval.save_oof(
        name, corpus, {args.seed: corrected.tolist()}, folds,
        notes=(
            "Out-of-fold predictions after group hate-share correction by "
            "instance reweighting. Scored against the ORIGINAL human labels."
        ),
    )

    # compare
    baseline = np.asarray(
        crossval.load_oof(args.model)["predictions"][str(args.seed)], dtype=int
    )
    y_true = corpus["label"].to_numpy()

    base_f1 = f1_score(y_true, baseline, average="macro", zero_division=0)
    corr_f1 = f1_score(y_true, corrected, average="macro", zero_division=0)
    base_acc = float((baseline == y_true).mean())
    corr_acc = float((corrected == y_true).mean())

    print("\n" + "=" * 92)
    print("COST OF THE CORRECTION  (out-of-fold, against original labels)")
    print("=" * 92)
    print(f"  {'':22s} {'baseline':>10s} {'corrected':>10s} {'change':>10s}")
    print(f"  {'macro-F1':22s} {base_f1:10.4f} {corr_f1:10.4f} "
          f"{corr_f1 - base_f1:+10.4f}")
    print(f"  {'accuracy':22s} {base_acc:10.4f} {corr_acc:10.4f} "
          f"{corr_acc - base_acc:+10.4f}")

    base_tbl = directional_table(corpus, baseline, task, groups)
    corr_tbl = directional_table(corpus, corrected, task, groups)

    print("\n" + "=" * 92)
    print("GROUP HATE-SHARE: HUMAN vs BASELINE vs CORRECTED")
    print("=" * 92)
    out = pd.DataFrame({
        "group": base_tbl["group"],
        "n": base_tbl["n_human_hostile"],
        "human": base_tbl["human_hate_share"].map(_pct),
        "baseline": base_tbl["model_hate_share"].map(_pct),
        "corrected": corr_tbl["model_hate_share"].map(_pct),
        "change": (
            corr_tbl["model_hate_share"] - base_tbl["model_hate_share"]
        ).map(lambda v: _pct(v, signed=True)),
    }).sort_values("human", ascending=False, key=lambda s: s.map(float))
    print(out.to_string(index=False))

    base_grad = gradient_agreement(base_tbl, config.MIN_GROUP_N)
    corr_grad = gradient_agreement(corr_tbl, config.MIN_GROUP_N)

    print("\n" + "=" * 92)
    print("DOES THE ORDERING SURVIVE?")
    print("=" * 92)
    print(f"  {'':22s} {'rho':>10s} {'p':>12s} {'mean shift':>12s}")
    for label, g in (("baseline", base_grad), ("corrected", corr_grad)):
        print(f"  {label:22s} {g['spearman_rho']:10.3f} {g['p_value']:12.2e} "
              f"{g['mean_shift']:+12.3f}")
    delta = corr_grad["spearman_rho"] - base_grad["spearman_rho"]
    print(f"  {'change in rho':22s} {delta:+10.3f}")

    # Rank correlation hides magnitude
    reliable = base_tbl["n_human_hostile"] >= config.MIN_GROUP_N
    spreads = {
        "human": float(base_tbl.loc[reliable, "human_hate_share"].max()
                       - base_tbl.loc[reliable, "human_hate_share"].min()),
        "baseline": float(base_tbl.loc[reliable, "model_hate_share"].max()
                          - base_tbl.loc[reliable, "model_hate_share"].min()),
        "corrected": float(corr_tbl.loc[reliable, "model_hate_share"].max()
                           - corr_tbl.loc[reliable, "model_hate_share"].min()),
    }
    removed = 1 - spreads["corrected"] / spreads["baseline"]
    print(f"\n  {'spread (max - min)':22s} {'human':>10s} {'baseline':>12s} "
          f"{'corrected':>12s}")
    print(f"  {'':22s} {_pct(spreads['human']):>9s}% "
          f"{_pct(spreads['baseline']):>11s}% {_pct(spreads['corrected']):>11s}%")
    print(f"  correction removed {100 * removed:.0f}% of the model's spread, "
          f"and {100 * (1 - corr_grad['spearman_rho'] / base_grad['spearman_rho']):.0f}%"
          " of its rank agreement")

    if corr_grad["spearman_rho"] > 0.7:
        verdict = (
            "ORDERING SURVIVES -- the gradient is not purely label-derived.\n"
            "  Equalising the labels did not remove it, so the models are\n"
            "  responding to something in the text that distinguishes these\n"
            "  communities' hostile content."
        )
    elif corr_grad["spearman_rho"] < 0.3:
        verdict = (
            "ORDERING COLLAPSES -- the gradient was learned from the labels.\n"
            "  With the annotators' differential standards removed from the\n"
            "  training signal, the models no longer rank the communities."
        )
    else:
        verdict = (
            "PARTIAL -- the ordering weakens but does not vanish. Both\n"
            "  mechanisms are contributing and neither can be dismissed."
        )
    print(f"\n  {verdict}")

    payload = {
        "model": args.model,
        "correction": report,
        "verification": check.to_dict(orient="records"),
        "cost": {
            "baseline_macro_f1": base_f1, "corrected_macro_f1": corr_f1,
            "macro_f1_change": corr_f1 - base_f1,
            "baseline_accuracy": base_acc, "corrected_accuracy": corr_acc,
        },
        "gradient": {"baseline": base_grad, "corrected": corr_grad},
        "spread": {**spreads, "fraction_of_model_spread_removed": removed},
        "per_group": out.to_dict(orient="records"),
    }
    path = config.RESULTS_DIR / f"label_correction_{args.model}.json"
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    out.to_csv(config.RESULTS_DIR / f"label_correction_{args.model}.csv",
               index=False)
    print(f"\nWritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
