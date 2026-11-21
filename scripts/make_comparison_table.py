from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from scripts.extract_false_positives import is_model_report
from src import config
from src.harness import (
    KEY_FUNCTIONALITIES,
    key_metrics,
    mcnemar_test,
    paired_bootstrap_diff,
)

# dissertation order
PARADIGM_ORDER = [
    "rule_based",
    "naive_bayes",
    "tfidf",
    "bilstm",
    "roberta",
    "llm",
    "guard",
]


def _paradigm_rank(name: str) -> int:
    for i, prefix in enumerate(PARADIGM_ORDER):
        if name.startswith(prefix):
            return i
    return len(PARADIGM_ORDER)


def load_reports(results_dir: Path) -> list[dict]:
    reports = []
    for path in sorted(results_dir.glob("*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        if is_model_report(blob):
            reports.append(blob)
    reports.sort(key=lambda r: (_paradigm_rank(r["model_name"]), r["model_name"]))
    return reports


def _key_metrics(report: dict) -> dict:
    if "key_metrics" in report:
        return report["key_metrics"]
    return key_metrics(
        report["slices"]["islam"],
        report["slices"]["non_islam"],
        report["hatecheck_muslim"],
    )


def group_severity_table(reports: list[dict]) -> pd.DataFrame | None:
    from src.config import ANALYSIS_GROUPS

    rows = []
    for group in ANALYSIS_GROUPS:
        row: dict = {"group": group}
        human_seen = False
        for r in reports:
            block = (r.get("target_groups") or {}).get(group)
            if not block:
                continue
            if not human_seen:
                row["n_hostile"] = block["human_hate_share"]["n"]
                row["human"] = block["human_hate_share"]["point"]
                human_seen = True
            row[r["model_name"]] = block["model_hate_share"]["point"]
        if human_seen:
            rows.append(row)

    if not rows:
        return None
    return pd.DataFrame(rows).sort_values(
        "human", ascending=False, na_position="last"
    ).reset_index(drop=True)


def gradient_table(reports: list[dict]) -> pd.DataFrame | None:
    rows = []
    for r in reports:
        agreement = (r.get("target_groups") or {}).get("_gradient_agreement")
        if not agreement or agreement.get("spearman_rho") is None:
            continue
        rows.append({
            "model": r["model_name"],
            "n_groups": agreement["n_groups"],
            "spearman_rho": agreement["spearman_rho"],
            "p_value": agreement.get("p_value"),
            "mean_shift": agreement.get("mean_shift"),
        })
    return pd.DataFrame(rows) if rows else None


def summary_table(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        km = _key_metrics(r)
        row = {"model": r["model_name"]}

        # 1. over-flagging: binary task only
        if km.get("over_flag_gap") is not None:
            row["overflag_islam"] = km["over_flag_islam"]
            row["overflag_rest"] = km["over_flag_rest"]
            row["overflag_gap"] = km["over_flag_gap"]

        # 1'. three-class equivalent: severity disagreement on the Islam slice
        islam = (r.get("target_groups") or {}).get("islam")
        if islam:
            human = islam["human_hate_share"]["point"]
            model = islam["model_hate_share"]["point"]
            row["islam_human_share"] = human
            row["islam_model_share"] = model
            row["islam_share_shift"] = (
                None if human is None or model is None else model - human
            )
            row["islam_escalation"] = islam["escalation_rate"]["point"]
            row["islam_de_escalation"] = islam["de_escalation_rate"]["point"]

        # 2. macro-F1: islam vs rest
        row["F1_islam"] = km["macro_f1_islam"]
        row["F1_rest"] = km["macro_f1_rest"]
        row["F1_gap"] = km["macro_f1_gap"]

        # 3. the four divergence functionalities, one column each
        for func in KEY_FUNCTIONALITIES:
            row[func] = km["hatecheck_key_functionalities"].get(func)
        if "runs" in r:
            row["n_seeds"] = r["runs"]["n_seeds"]
        rows.append(row)
    return pd.DataFrame(rows)


def summary_variability_table(reports: list[dict]) -> pd.DataFrame | None:
    rows = []
    for r in reports:
        if "runs" not in r:
            continue
        # std mirrors the report structure
        std_km = r["runs"]["std"]["key_metrics"]
        std_head = r["runs"]["std"]["headline"]
        km = _key_metrics(r)
        row = {
            "model": r["model_name"],
            "n_seeds": r["runs"]["n_seeds"],
            "seeds": ", ".join(str(s) for s in r["runs"]["seeds"]),
            "macro_F1": f"{r['headline']['macro_f1']:.4f} ± {std_head['macro_f1']:.4f}",
            "F1_gap": f"{km['macro_f1_gap']:+.4f} ± {std_km['macro_f1_gap']:.4f}",
        }
        if km.get("over_flag_gap") is not None:
            row["overflag_gap"] = (
                f"{km['over_flag_gap']:+.4f} ± {std_km['over_flag_gap']:.4f}"
            )
        islam_std = (r["runs"]["std"].get("target_groups") or {}).get("islam")
        islam = (r.get("target_groups") or {}).get("islam")
        if islam and islam_std:
            row["islam_model_share"] = (
                f"{islam['model_hate_share']['point']:.4f} "
                f"± {islam_std['model_hate_share']['point']:.4f}"
            )
        for func in KEY_FUNCTIONALITIES:
            mean = km["hatecheck_key_functionalities"][func]
            std_f = std_km["hatecheck_key_functionalities"][func]
            row[func] = f"{mean:.4f} ± {std_f:.4f}"
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def headline_table(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        h = r["headline"]
        ci = r["confidence_intervals"]["headline_macro_f1"]
        row = {
            "model": r["model_name"],
            "accuracy": h["accuracy"],
            "macro_F1": h["macro_f1"],
            "macro_F1_95CI": f"[{ci['lo']:.3f}, {ci['hi']:.3f}]",
        }
        # per-class F1 for whatever classes this task has
        for cls, m in h["per_class"].items():
            row[f"{cls}_P"] = m["precision"]
            row[f"{cls}_R"] = m["recall"]
            row[f"{cls}_F1"] = m["f1"]
        if "over_flag_rate" in h:
            row["over_flag"] = h["over_flag_rate"]
            row["under_detect"] = h["under_detect_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


def slice_table(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        isl = r["slices"]["islam"]
        rest = r["slices"]["non_islam"]
        row = {
            "model": r["model_name"],
            "islam_n": isl["n"],
            "islam_macro_F1": isl["macro_f1"],
            "rest_macro_F1": rest["macro_f1"],
            "gap_macro_F1": isl["macro_f1"] - rest["macro_f1"],
        }
        for cls, m in isl["per_class"].items():
            row[f"islam_{cls}_R"] = m["recall"]
        if "over_flag_rate" in isl:
            row["islam_over_flag"] = isl["over_flag_rate"]
            row["islam_under_detect"] = isl["under_detect_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


def hatecheck_table(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        hc = r["hatecheck_muslim"]
        rows.append({
            "model": r["model_name"],
            "n_cases": hc["n_cases"],
            "recall_373_hateful": hc["recall_on_hateful"],
            "false_alarm_111_nonhateful": hc["false_alarm_on_non_hateful"],
            "accuracy": hc["accuracy"],
        })
    return pd.DataFrame(rows)


def functionality_table(reports: list[dict]) -> pd.DataFrame:
    frames = {}
    gold = {}
    for r in reports:
        per = r["hatecheck_muslim"]["per_functionality"]
        frames[r["model_name"]] = {f: m["pass_rate"] for f, m in per.items()}
        gold.update({f: m["gold"] for f, m in per.items()})
    df = pd.DataFrame(frames)
    df.insert(0, "gold", pd.Series(gold))
    return df.sort_values(["gold", df.columns[1]]).reset_index(names="functionality")


def cost_table(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        c = r["cost"]
        size = c.get("model_size_bytes")
        rows.append({
            "model": r["model_name"],
            "train_s": c["train_seconds"],
            "tune_s": c.get("val_tune_seconds", 0.0),
            "infer_s_per_1000": c["inference_seconds_per_1000"],
            "size_MB": round(size / 1e6, 3) if size else None,
            "api_cost_usd": c.get("api_cost_usd", 0.0),
        })
    return pd.DataFrame(rows)


def pairwise_mcnemar(reports: list[dict], n_bootstrap: int = 2000) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(reports):
        for b in reports[i + 1:]:
            if a["predictions"]["post_ids"] != b["predictions"]["post_ids"]:
                raise ValueError(
                    f"{a['model_name']} and {b['model_name']} were scored on "
                    "different test rows -- rebuild the splits and rerun"
                )
            y_true = a["predictions"]["y_true"]
            pred_a = a["predictions"]["y_pred"]
            pred_b = b["predictions"]["y_pred"]

            res = mcnemar_test(y_true, pred_a, pred_b)
            ci = paired_bootstrap_diff(
                y_true, pred_a, pred_b, "macro_f1", n_resamples=n_bootstrap
            )
            rows.append({
                "model_a": a["model_name"],
                "model_b": b["model_name"],
                "acc_a": res["accuracy_a"],
                "acc_b": res["accuracy_b"],
                "macroF1_diff": ci["point"],
                "diff_95CI": f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}]",
                "CI_width": ci["width"],
                "CI_excludes_0": ci["excludes_zero"],
                "mcnemar_p": res["p_value"],
                "sig_at_0.05": res["p_value"] < 0.05,
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--no-mcnemar", action="store_true")
    args = parser.parse_args()

    reports = load_reports(args.results_dir)
    if not reports:
        print(f"No results found in {args.results_dir}. Run a model first.")
        return 1

    tables: dict[str, pd.DataFrame] = {}

    # Headline result leads
    groups = group_severity_table(reports)
    if groups is not None:
        tables["HEADLINE -- hate-share by target group (human vs each model)"] = groups
    gradient = gradient_table(reports)
    if gradient is not None:
        tables["Rank agreement with the human severity gradient"] = gradient

    tables.update({
        "Per-slice gaps and the four divergence functionalities":
            summary_table(reports),
        "Overall performance (HateXplain test split)": headline_table(reports),
        "Worked slice: Islam-targeted vs rest (one community of the ten)":
            slice_table(reports),
        "HateCheck (Muslim-targeted subset, the only one at usable size)":
            hatecheck_table(reports),
        "HateCheck per-functionality pass rate": functionality_table(reports),
        "Cost": cost_table(reports),
    })
    variability = summary_variability_table(reports)
    if variability is not None:
        tables["Run-to-run variability (multi-seed models, mean ± std)"] = variability
    if not args.no_mcnemar and len(reports) > 1:
        tables["Paired comparison: bootstrap CI on macro-F1 difference + McNemar"] = (
            pairwise_mcnemar(reports)
        )

    md_parts = [
        "# Cross-model comparison",
        "",
        f"Models: {len(reports)} | seed: {reports[0]['seed']} | "
        f"task: {reports[0]['task']['name']} "
        f"({reports[0]['task']['definition']})",
        "",
    ]
    # Captions travel with the table
    notes = {
        "HEADLINE -- hate-share by target group (human vs each model)": (
            "  Held-out test split only, so most communities fall below the "
            "n>=200 reliability bar.\n"
            "  The properly powered version -- all ten communities on "
            "out-of-fold predictions over\n  the full corpus, plus the "
            "label-free LLM row -- is scripts/analyse_target_groups.py."
        ),
        "Rank agreement with the human severity gradient": (
            "  Read rho next to macro-F1 below: a model that cannot grade "
            "severity produces a flat\n  profile, and a flat profile is "
            "uncorrelated with any gradient. See 'Flat is not fair'."
        ),
    }

    for title, df in tables.items():
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        note = notes.get(title)
        if note:
            print(f"\n{note}")
        md_parts += [f"## {title}", ""]
        if note:
            md_parts += [note.replace("  ", "", 1).strip(), ""]
        md_parts += [df.to_markdown(index=False, floatfmt=".4f"), ""]

    md_path = args.results_dir / "comparison.md"
    md_path.write_text("\n".join(md_parts), encoding="utf-8")

    # One flat CSV
    csv_path = args.results_dir / "comparison.csv"
    summary_table(reports).merge(
        headline_table(reports).drop(columns=["macro_F1_95CI"]), on="model"
    ).merge(
        hatecheck_table(reports).drop(columns=["n_cases"]),
        on="model", suffixes=("", "_hatecheck"),
    ).to_csv(csv_path, index=False)

    print(f"\nWritten: {md_path}\n         {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
