from __future__ import annotations

import pytest

from scripts.make_comparison_table import (
    headline_table,
    pairwise_mcnemar,
    summary_table,
    summary_variability_table,
)
from src.harness import KEY_FUNCTIONALITIES, aggregate_reports


def _report(name: str, seed: int = 42, acc: float = 0.8, gap: float = 0.1) -> dict:
    return {
        "model_name": name,
        "seed": seed,
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "task": {"name": "hate_only"},
        "split": {"seed": 42},
        "environment": {"python": "3.13"},
        "params": {},
        "notes": "",
        "headline": {
            "n": 4,
            "accuracy": acc,
            "macro_f1": acc - 0.02,
            "per_class": {
                "not_hate": {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 2},
                "hate": {"precision": 0.7, "recall": 0.7, "f1": 0.7, "support": 2},
            },
            "confusion_matrix": {"tn": 1, "fp": 1, "fn": 1, "tp": 1},
            "over_flag_rate": 0.2,
            "under_detect_rate": 0.3,
        },
        "slices": {
            "islam": {"n": 2, "macro_f1": 0.7, "over_flag_rate": 0.2 + gap,
                      "under_detect_rate": 0.3,
                      "per_class": {"hate": {"recall": 0.7}}},
            "non_islam": {"n": 2, "macro_f1": 0.8, "over_flag_rate": 0.2,
                          "under_detect_rate": 0.3,
                          "per_class": {"hate": {"recall": 0.7}}},
        },
        "hatecheck_muslim": {
            "n_cases": 4, "recall_on_hateful": 0.5,
            "false_alarm_on_non_hateful": 0.3, "accuracy": 0.6,
            "per_functionality": {
                f: {"n": 1, "gold": "non-hateful", "n_correct": 1,
                    "pass_rate": 0.5 + 0.1 * i, "flag_rate": 0.5}
                for i, f in enumerate(KEY_FUNCTIONALITIES)
            },
        },
        "key_metrics": {
            "over_flag_islam": 0.2 + gap, "over_flag_rest": 0.2,
            "over_flag_gap": gap,
            "macro_f1_islam": 0.7, "macro_f1_rest": 0.8, "macro_f1_gap": -0.1,
            "hatecheck_key_functionalities": {
                f: 0.5 + 0.1 * i for i, f in enumerate(KEY_FUNCTIONALITIES)
            },
        },
        "confidence_intervals": {
            "headline_macro_f1": {"lo": 0.7, "hi": 0.9, "point": 0.8},
        },
        "cost": {"train_seconds": 1.0, "val_tune_seconds": 0.0,
                 "inference_seconds_per_1000": 0.1, "model_size_bytes": 1000},
        "predictions": {"post_ids": ["a", "b"], "y_true": [1, 0], "y_pred": [1, 1]},
        "hatecheck_predictions": {"case_ids": ["1"], "y_true": [1], "y_pred": [1]},
    }


def test_summary_table_has_the_three_promoted_metrics():
    df = summary_table([_report("m1")])
    for col in ("overflag_islam", "overflag_rest", "overflag_gap",
                "F1_islam", "F1_rest", "F1_gap"):
        assert col in df.columns
    for func in KEY_FUNCTIONALITIES:
        assert func in df.columns, f"{func} must be its own column"


def test_summary_table_columns_are_in_reading_order():
    cols = list(summary_table([_report("m1")]).columns)
    assert cols[0] == "model"
    assert cols[1:4] == ["overflag_islam", "overflag_rest", "overflag_gap"]
    assert cols[4:7] == ["F1_islam", "F1_rest", "F1_gap"]
    assert cols[7:11] == list(KEY_FUNCTIONALITIES)


def test_summary_table_one_row_per_model():
    df = summary_table([_report("m1"), _report("m2")])
    assert list(df["model"]) == ["m1", "m2"]


def test_summary_table_recomputes_key_metrics_for_older_reports():
    old = _report("old")
    del old["key_metrics"]
    df = summary_table([old])
    assert df.loc[0, "F1_gap"] == pytest.approx(-0.1)


def test_variability_table_is_none_without_multi_seed_models():
    assert summary_variability_table([_report("m1")]) is None


def test_variability_table_formats_mean_and_std():
    runs = [_report(f"m_seed{s}", seed=s, acc=a, gap=g)
            for s, a, g in [(42, 0.80, 0.10), (43, 0.84, 0.14)]]
    agg = aggregate_reports(runs, "m", write=False)
    df = summary_variability_table([agg])
    assert df is not None
    assert df.loc[0, "n_seeds"] == 2
    assert "±" in df.loc[0, "overflag_gap"]
    assert df.loc[0, "overflag_gap"].startswith("+0.1200")


def test_summary_table_flags_multi_seed_models():
    runs = [_report(f"m_seed{s}", seed=s) for s in (42, 43)]
    agg = aggregate_reports(runs, "m", write=False)
    df = summary_table([_report("single"), agg])
    assert df.loc[1, "n_seeds"] == 2


def test_mcnemar_refuses_models_scored_on_different_rows():
    a, b = _report("a"), _report("b")
    b["predictions"]["post_ids"] = ["x", "y"]
    with pytest.raises(ValueError, match="different test rows"):
        pairwise_mcnemar([a, b])


def test_headline_table_carries_the_confidence_interval():
    df = headline_table([_report("m1")])
    assert df.loc[0, "macro_F1_95CI"] == "[0.700, 0.900]"
