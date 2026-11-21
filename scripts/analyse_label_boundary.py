from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from scripts.extract_false_positives import is_model_report
from src import config, data
from src.harness import over_flag_by_original_label


def load_reports(results_dir: Path) -> list[dict]:
    reports = []
    for path in sorted(results_dir.glob("*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        if is_model_report(blob):
            reports.append(blob)
    return reports


def build_table(reports: list[dict], test: pd.DataFrame) -> pd.DataFrame:
    indexed = test.set_index("post_id")
    rows = []
    for report in reports:
        preds = report["predictions"]
        # align by post_id, not order
        sub = indexed.loc[preds["post_ids"]]
        breakdown = over_flag_by_original_label(
            preds["y_true"],
            preds["y_pred"],
            sub["majority_label"].tolist(),
            sub["is_islam_slice"].tolist(),
        )
        row = {"model": report["model_name"]}
        for bucket in ("normal", "offensive"):
            b = breakdown[bucket]
            row[f"{bucket}_islam"] = b["islam"]["over_flag_rate"]
            row[f"{bucket}_rest"] = b["rest"]["over_flag_rate"]
            row[f"{bucket}_gap"] = b["gap"]
            row[f"{bucket}_n_islam"] = b["islam"]["n"]
            row[f"{bucket}_n_flagged_islam"] = b["islam"]["n_flagged"]

        # composition versus behaviour
        n_norm_i = breakdown["normal"]["islam"]["n"]
        n_off_i = breakdown["offensive"]["islam"]["n"]
        w_norm = n_norm_i / (n_norm_i + n_off_i)
        w_off = 1 - w_norm
        pooled_islam = (
            breakdown["normal"]["islam"]["n_flagged"]
            + breakdown["offensive"]["islam"]["n_flagged"]
        ) / (n_norm_i + n_off_i)
        n_norm_r = breakdown["normal"]["rest"]["n"]
        n_off_r = breakdown["offensive"]["rest"]["n"]
        pooled_rest = (
            breakdown["normal"]["rest"]["n_flagged"]
            + breakdown["offensive"]["rest"]["n_flagged"]
        ) / (n_norm_r + n_off_r)
        # Direct standardisation
        standardised_rest = (
            w_norm * breakdown["normal"]["rest"]["over_flag_rate"]
            + w_off * breakdown["offensive"]["rest"]["over_flag_rate"]
        )
        row["pooled_gap"] = pooled_islam - pooled_rest
        row["standardised_gap"] = pooled_islam - standardised_rest
        row["composition_effect"] = standardised_rest - pooled_rest

        # -- is the normal-bucket gap distinguishable from zero at all?
        from scipy.stats import fisher_exact

        table = [
            [breakdown["normal"]["islam"]["n_flagged"],
             n_norm_i - breakdown["normal"]["islam"]["n_flagged"]],
            [breakdown["normal"]["rest"]["n_flagged"],
             n_norm_r - breakdown["normal"]["rest"]["n_flagged"]],
        ]
        row["normal_fisher_p"] = float(fisher_exact(table).pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_BINARY_DIR)
    args = parser.parse_args()

    reports = load_reports(args.results_dir)
    if not reports:
        print(f"No results in {args.results_dir}")
        return 1

    # Binary task only
    wrong = sorted(
        {r["task"]["name"] for r in reports if r["task"]["name"] != "hate_only"}
    )
    if wrong:
        print(f"{args.results_dir} holds {', '.join(wrong)} results -- "
              "this analysis is binary only.")
        print("Run it against the archive instead:")
        print(f"    python -m scripts.analyse_label_boundary "
              f"--results-dir {config.RESULTS_BINARY_DIR}")
        return 2

    # Binary reports need binary data
    _, _, test = data.get_dataset(task=config.TASK_BINARY)
    table = build_table(reports, test)

    counts = test[test["label"] == 0].groupby(
        ["is_islam_slice", "majority_label"]
    ).size()
    print("Test-split non-hateful posts by bucket:")
    print(counts.to_string())

    print("\n" + "=" * 88)
    print("NORMAL -> hate   (real false positives)")
    print("=" * 88)
    print(
        table[["model", "normal_islam", "normal_rest", "normal_gap",
               "normal_n_islam", "normal_n_flagged_islam"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    print("\n" + "=" * 88)
    print("OFFENSIVE -> hate   (label-boundary disagreement)")
    print("=" * 88)
    print(
        table[["model", "offensive_islam", "offensive_rest", "offensive_gap",
               "offensive_n_islam", "offensive_n_flagged_islam"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    print("\n" + "=" * 88)
    print("Composition vs behaviour")
    print("=" * 88)
    print("slice 67% offensive vs rest 42%; 'standardised' re-weights to match")
    print(
        table[["model", "pooled_gap", "composition_effect", "standardised_gap"]]
        .to_string(index=False, float_format=lambda v: f"{v:+.4f}")
    )

    print("\n" + "=" * 88)
    print("Does the gap survive in the normal bucket alone?")
    print("=" * 88)
    for _, r in table.iterrows():
        p = r["normal_fisher_p"]
        if p >= 0.05:
            verdict = f"NO  -- normal-bucket gap not distinguishable (Fisher p={p:.2f})"
        elif r["normal_gap"] > 0:
            verdict = f"YES -- real over-flagging (Fisher p={p:.3f})"
        else:
            verdict = f"reversed -- flags fewer (Fisher p={p:.3f})"
        print(
            f"  {r['model']:28s} normal {r['normal_gap']:+.4f}  "
            f"offensive {r['offensive_gap']:+.4f}   {verdict}"
        )
    print(f"\n  caveat: only {int(table.iloc[0]['normal_n_islam'])} Islam "
          "'normal' posts in test -- underpowered")

    out = args.results_dir / "label_boundary_breakdown.csv"
    table.to_csv(out, index=False)
    print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
