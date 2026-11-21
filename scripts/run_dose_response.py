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
from src.reweight import group_balance_weights

# Capability floor
CAPABILITY_FLOOR = 0.50

# Spread floor
SPREAD_FLOOR = 0.15

# Frozen needs a probe rate
CONDITIONS: dict[str, dict] = {
    "frozen": {
        "freeze_encoder": True, "train_fraction": 1.0, "lr": 1e-3,
        "label": "frozen encoder, head only",
        "exposure": "head only",
    },
    "25pct": {
        "freeze_encoder": False, "train_fraction": 0.25, "lr": 2e-5,
        "label": "fine-tuned, 25% of train",
        "exposure": "25%",
    },
    "50pct": {
        "freeze_encoder": False, "train_fraction": 0.50, "lr": 2e-5,
        "label": "fine-tuned, 50% of train",
        "exposure": "50%",
    },
    "full": {
        "freeze_encoder": False, "train_fraction": 1.00, "lr": 2e-5,
        "label": "fine-tuned, 100% of train",
        "exposure": "100%",
        "reuse": ("roberta_base", "roberta_base_corrected"),
    },
}


def oof_for(name: str, corpus, folds, weights, builder, seed: int, progress):
    try:
        blob = crossval.load_oof(name)
        progress(f"  reusing {name}")
        return np.asarray(blob["predictions"][str(blob["seeds"][0])], dtype=int)
    except FileNotFoundError:
        pass
    preds = crossval.out_of_fold_predictions(
        corpus, folds, builder, seed=seed, sample_weight=weights,
        progress=progress,
    )
    crossval.save_oof(
        name, corpus, {seed: preds.tolist()}, folds,
        notes="Dose-response condition; see scripts/run_dose_response.py.",
    )
    return preds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    from src.models.transformer import TransformerDetector, require_torch

    require_torch()

    task = config.DEFAULT_TASK
    corpus = data.labelled_hatexplain(task=task).sort_values(
        "post_id", kind="mergesort"
    ).reset_index(drop=True)
    folds = crossval.load_folds()
    groups = config.ANALYSIS_GROUPS
    y = corpus["label"].to_numpy()

    weights, wreport = group_balance_weights(corpus, groups)
    print("=" * 100)
    print("DOSE-RESPONSE: label-dependence against corpus exposure")
    print("=" * 100)
    print(f"  architecture   roberta-base, identical across all conditions")
    print(f"  folds          {folds.n_splits}, seed {args.seed}")
    print(f"  correction     IPF to {100 * wreport['target_hate_share']:.1f}% "
          f"hate-share, converged to {wreport['ipf_max_deviation']:.1e}")
    print(f"  floors         macro-F1 >= {CAPABILITY_FLOOR}, "
          f"group spread >= {SPREAD_FLOOR}")

    rows = []
    for key in args.conditions:
        cfg = CONDITIONS[key]
        print(f"\n{'#' * 100}\n# {key}: {cfg['label']}  (lr {cfg['lr']})\n{'#' * 100}")
        started = time.perf_counter()

        def make(corrected: bool):
            def build(fold, seed):
                return TransformerDetector(
                    name=f"dose_{key}{'_corrected' if corrected else ''}_fold{fold}",
                    seed=seed,
                    epochs=args.epochs,
                    lr=cfg["lr"],
                    freeze_encoder=cfg["freeze_encoder"],
                    train_fraction=cfg["train_fraction"],
                )
            return build

        base_name, corr_name = cfg.get(
            "reuse", (f"dose_{key}", f"dose_{key}_corrected")
        )
        baseline = oof_for(base_name, corpus, folds, None, make(False),
                           args.seed, print)
        corrected = oof_for(corr_name, corpus, folds, weights, make(True),
                            args.seed, print)
        elapsed = time.perf_counter() - started

        base_f1 = f1_score(y, baseline, average="macro", zero_division=0)
        corr_f1 = f1_score(y, corrected, average="macro", zero_division=0)
        base_tbl = directional_table(corpus, baseline, task, groups)
        corr_tbl = directional_table(corpus, corrected, task, groups)
        bg = gradient_agreement(base_tbl, config.MIN_GROUP_N)
        cg = gradient_agreement(corr_tbl, config.MIN_GROUP_N)

        reliable = base_tbl["n_human_hostile"] >= config.MIN_GROUP_N
        base_spread = float(
            base_tbl.loc[reliable, "model_hate_share"].max()
            - base_tbl.loc[reliable, "model_hate_share"].min()
        )
        corr_spread = float(
            corr_tbl.loc[reliable, "model_hate_share"].max()
            - corr_tbl.loc[reliable, "model_hate_share"].min()
        )

        interpretable = base_f1 >= CAPABILITY_FLOOR and base_spread >= SPREAD_FLOOR
        rows.append({
            "condition": key,
            "exposure": cfg["exposure"],
            "label": cfg["label"],
            "lr": cfg["lr"],
            "macro_f1": base_f1,
            "baseline_rho": bg["spearman_rho"],
            "corrected_rho": cg["spearman_rho"],
            "delta_rho": (cg["spearman_rho"] - bg["spearman_rho"]),
            "rho_lost_pct": 100 * (1 - cg["spearman_rho"] / bg["spearman_rho"])
            if bg["spearman_rho"] else None,
            "base_spread": base_spread,
            "spread_removed_pct": 100 * (1 - corr_spread / base_spread)
            if base_spread else None,
            "macro_f1_cost": corr_f1 - base_f1,
            "interpretable": interpretable,
            "minutes": elapsed / 60,
        })
        r = rows[-1]
        print(f"\n  macro-F1 {base_f1:.4f} | rho {bg['spearman_rho']:+.3f} -> "
              f"{cg['spearman_rho']:+.3f} ({r['delta_rho']:+.3f}) | "
              f"spread removed {r['spread_removed_pct']:.0f}% | "
              f"{elapsed / 60:.1f} min")
        if not interpretable:
            print(f"  BELOW FLOOR (macro-F1 {base_f1:.3f}, spread "
                  f"{base_spread:.3f}) -- label-dependence uninterpretable")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    show = df[["condition", "exposure", "macro_f1", "baseline_rho",
               "corrected_rho", "delta_rho", "rho_lost_pct",
               "spread_removed_pct", "macro_f1_cost", "interpretable"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    usable = df[df["interpretable"]]
    excluded = df[~df["interpretable"]]
    if len(excluded):
        print(f"\n  excluded as below the capability floor: "
              f"{', '.join(excluded['condition'])}")

    print("\n" + "=" * 100)
    print("DOES LABEL-DEPENDENCE TRACK CORPUS EXPOSURE?")
    print("=" * 100)
    if len(usable) < 3:
        print(f"  Only {len(usable)} interpretable conditions -- not enough to "
              "read a trend.")
    else:
        order = [k for k in CONDITIONS if k in set(usable["condition"])]
        seq = usable.set_index("condition").loc[order]
        for measure in ("spread_removed_pct", "rho_lost_pct"):
            vals = seq[measure].tolist()
            rising = all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))
            falling = all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))
            arrow = " -> ".join(f"{v:.0f}%" for v in vals)
            verdict = ("MONOTONIC RISING" if rising else
                       "MONOTONIC FALLING" if falling else "NOT MONOTONIC")
            print(f"  {measure:20s} {arrow}   {verdict}")
        print("\n  predicted: monotonic rising in both")

    out = config.RESULTS_DIR / "dose_response.json"
    out.write_text(json.dumps({
        "capability_floor": CAPABILITY_FLOOR,
        "spread_floor": SPREAD_FLOOR,
        "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "reuse"}
                       for k, v in CONDITIONS.items()},
        "correction": wreport,
        "results": rows,
    }, indent=2, default=float), encoding="utf-8")
    df.to_csv(config.RESULTS_DIR / "dose_response.csv", index=False)
    print(f"\nWritten: {out}")

    try:
        from scripts.plot_dose_response import plot_dose_response

        path = plot_dose_response(df, config.RESULTS_DIR)
        print(f"Plot written: {path}")
    except ImportError as exc:
        print(f"(plot skipped: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
