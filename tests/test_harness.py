from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import TASK_BINARY, TASK_THREE_CLASS
from src.harness import (
    BaseDetector,
    Detector,
    EvalInputs,
    bootstrap_ci,
    classification_metrics,
    evaluate,
    hatecheck_report,
    mcnemar_test,
)


# metric blocks


BINARY = TASK_BINARY.class_names
THREE = TASK_THREE_CLASS.class_names


def test_perfect_predictions():
    m = classification_metrics([0, 1, 0, 1], [0, 1, 0, 1], BINARY)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["over_flag_rate"] == 0.0
    assert m["under_detect_rate"] == 0.0


def test_confusion_matrix_orientation():
    # two negatives called positive, one positive missed
    m = classification_metrics([0, 0, 1, 1], [1, 1, 0, 1], BINARY)
    cm = m["confusion_matrix"]
    assert (cm["tn"], cm["fp"], cm["fn"], cm["tp"]) == (0, 2, 1, 1)
    assert m["over_flag_rate"] == 1.0        # 2 of 2 negatives flagged
    assert m["under_detect_rate"] == 0.5     # 1 of 2 positives missed


def test_degenerate_all_negative_still_reports_2x2():
    m = classification_metrics([0, 1, 1], [0, 0, 0], BINARY)
    assert np.array(m["confusion_matrix"]["matrix"]).shape == (2, 2)
    assert m["per_class"]["hate"]["recall"] == 0.0
    assert m["under_detect_rate"] == 1.0


# three-class


def test_three_class_confusion_matrix_is_3x3():
    m = classification_metrics([0, 1, 2, 0], [0, 2, 2, 1], THREE)
    assert np.array(m["confusion_matrix"]["matrix"]).shape == (3, 3)
    assert m["confusion_matrix"]["labels"] == list(THREE)
    assert set(m["per_class"]) == set(THREE)


def test_three_class_confusion_rows_are_truth():
    # one hatespeech post predicted offensive
    m = classification_metrics([0], [1], THREE)
    matrix = m["confusion_matrix"]["matrix"]
    assert matrix[0][1] == 1
    assert matrix[1][0] == 0


def test_binary_only_rates_are_absent_for_three_class():
    m = classification_metrics([0, 1, 2], [0, 1, 2], THREE)
    assert "over_flag_rate" not in m
    assert "under_detect_rate" not in m


def test_degenerate_three_class_model_still_reports_3x3():
    m = classification_metrics([0, 1, 2], [2, 2, 2], THREE)
    assert np.array(m["confusion_matrix"]["matrix"]).shape == (3, 3)
    assert m["per_class"]["hatespeech"]["recall"] == 0.0
    assert m["per_class"]["normal"]["recall"] == 1.0


def test_per_class_bootstrap_works_for_any_class_index():
    y = [0, 1, 2] * 30
    p = [0, 1, 2] * 30
    for i in range(3):
        ci = bootstrap_ci(y, p, f"recall_{i}", n_resamples=100)
        assert ci["point"] == pytest.approx(1.0)


def test_metrics_are_json_safe():
    import json

    json.dumps(classification_metrics([0, 1], [1, 0], BINARY))
    json.dumps(classification_metrics([0, 1, 2], [1, 0, 2], THREE))


# statistics


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    pred = np.where(rng.random(400) < 0.85, y, 1 - y)
    ci = bootstrap_ci(y, pred, "accuracy", n_resamples=500)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert 0.0 <= ci["lo"] and ci["hi"] <= 1.0


def test_bootstrap_ci_is_seeded():
    y = [0, 1] * 50
    p = [0, 1] * 40 + [1, 0] * 10
    a = bootstrap_ci(y, p, "macro_f1", n_resamples=200, seed=7)
    b = bootstrap_ci(y, p, "macro_f1", n_resamples=200, seed=7)
    assert a == b


def test_mcnemar_identical_models_are_not_significant():
    y = [0, 1] * 20
    res = mcnemar_test(y, y, y)
    assert res["p_value"] == 1.0
    assert res["n01"] == 0 if "n01" in res else True


def test_mcnemar_detects_a_clear_difference():
    y = [1] * 100
    good = [1] * 100          # all correct
    bad = [0] * 60 + [1] * 40  # 60 wrong
    res = mcnemar_test(y, good, bad)
    assert res["n01_a_correct_b_wrong"] == 60
    assert res["n10_a_wrong_b_correct"] == 0
    assert res["p_value"] < 0.001


def test_mcnemar_uses_exact_test_when_discordant_count_is_small():
    y = [1] * 100
    a = [1] * 100
    b = [0] * 5 + [1] * 95
    res = mcnemar_test(y, a, b)
    assert res["method"] == "exact binomial"


# HateCheck report


def _toy_hatecheck() -> pd.DataFrame:
    return pd.DataFrame({
        "case_id": [1, 2, 3, 4],
        "functionality": ["derog_h", "derog_h", "ident_pos_nh", "ident_pos_nh"],
        "test_case": ["a", "b", "c", "d"],
        "label_gold": ["hateful", "hateful", "non-hateful", "non-hateful"],
        "label": [1, 1, 0, 0],
    })


def test_hatecheck_pass_rate_reads_both_directions():
    df = _toy_hatecheck()
    rep = hatecheck_report(df, [1, 0, 1, 0], TASK_BINARY)
    assert rep["recall_on_hateful"] == 0.5
    assert rep["false_alarm_on_non_hateful"] == 0.5
    # for a hateful functionality pass rate == recall
    assert rep["per_functionality"]["derog_h"]["pass_rate"] == 0.5
    # for a non-hateful one it is 1 - flag rate
    assert rep["per_functionality"]["ident_pos_nh"]["pass_rate"] == 0.5
    assert rep["per_functionality"]["ident_pos_nh"]["flag_rate"] == 0.5


def test_hatecheck_strict_and_lenient_mappings_for_three_class():
    df = _toy_hatecheck()
    # hateful cases predicted hatespeech and offensive; nh cases predicted normal
    rep = hatecheck_report(df, [0, 1, 2, 2], TASK_THREE_CLASS)
    assert rep["recall_on_hateful"] == 0.5           # only the hatespeech one
    assert rep["lenient"]["recall_on_hateful"] == 1.0
    assert rep["downgraded_share_of_hateful"] == 0.5
    assert rep["false_alarm_on_non_hateful"] == 0.0


def test_to_hatecheck_binary_is_identity_for_a_binary_task():
    from src.harness import to_hatecheck_binary

    out = to_hatecheck_binary([0, 1, 1, 0], TASK_BINARY)
    assert out.tolist() == [0, 1, 1, 0]


# the Detector contract


class _AlwaysHate(BaseDetector):
    name = "always_hate"

    def predict(self, texts):
        return [1] * len(texts)


class _CountingDetector(BaseDetector):

    name = "counting"

    def __init__(self):
        self.seen: list[str] = []

    def fit(self, texts, labels):
        self.seen.extend(texts)

    def predict(self, texts):
        return [0] * len(texts)


def test_base_detector_satisfies_the_protocol():
    assert isinstance(_AlwaysHate(), Detector)


def _toy_inputs() -> EvalInputs:
    frame = pd.DataFrame({
        "post_id": [f"p{i}" for i in range(8)],
        "text": [f"text {i}" for i in range(8)],
        "label": [0, 1] * 4,
        "majority_label": ["normal", "hatespeech"] * 4,
        "target_groups": [("islam",), ("islam",), (), ()] * 2,
        "is_islam_slice": [True, True, False, False] * 2,
    })
    return EvalInputs(
        train=frame.iloc[:4], val=frame.iloc[4:6], test=frame.iloc[6:],
        hatecheck=_toy_hatecheck(), task=TASK_BINARY,
    )


def _toy_three_class_inputs() -> EvalInputs:
    labels = ["hatespeech", "offensive", "normal"]
    frame = pd.DataFrame({
        "post_id": [f"p{i}" for i in range(12)],
        "text": [f"text {i}" for i in range(12)],
        "label": [0, 1, 2] * 4,
        "majority_label": labels * 4,
        # islam in every split
        "target_groups": [("islam",), ("islam",), ("jewish",)] * 4,
        "is_islam_slice": [True, True, False] * 4,
    })
    return EvalInputs(
        train=frame.iloc[:6], val=frame.iloc[6:9], test=frame.iloc[9:],
        hatecheck=_toy_hatecheck(), task=TASK_THREE_CLASS,
    )


def test_evaluate_produces_every_required_section():
    report = evaluate(_AlwaysHate(), _toy_inputs(), n_bootstrap=50, write=False)
    for key in (
        "headline", "slices", "hatecheck_muslim", "cost",
        "confidence_intervals", "predictions", "seed", "task",
    ):
        assert key in report
    assert "islam" in report["slices"] and "non_islam" in report["slices"]
    assert report["cost"]["train_seconds"] >= 0.0


def test_evaluate_never_shows_the_test_split_to_the_model():
    inputs = _toy_inputs()
    det = _CountingDetector()
    evaluate(det, inputs, n_bootstrap=20, write=False)
    assert set(det.seen) == set(inputs.train["text"])


def test_evaluate_rejects_a_detector_that_returns_the_wrong_length():
    class _Broken(BaseDetector):
        name = "broken"

        def predict(self, texts):
            return [1]

    with pytest.raises(ValueError, match="returned 1 labels"):
        evaluate(_Broken(), _toy_inputs(), n_bootstrap=10, write=False)


def test_report_is_json_serialisable():
    import json

    report = evaluate(_AlwaysHate(), _toy_inputs(), n_bootstrap=20, write=False)
    json.dumps(report)


def test_three_class_report_carries_the_group_block():
    class _AlwaysHateSpeech(BaseDetector):
        name = "always_hatespeech"

        def predict(self, texts):
            return [0] * len(texts)

    report = evaluate(
        _AlwaysHateSpeech(), _toy_three_class_inputs(), n_bootstrap=20, write=False
    )
    assert report["task"]["n_classes"] == 3
    assert report["target_groups"]
    assert "islam" in report["target_groups"]
    # calling everything hatespeech means a hate-share of 1.0
    assert report["target_groups"]["islam"]["model_hate_share"]["point"] == 1.0
    # and no de-escalation at all
    assert report["target_groups"]["islam"]["de_escalation_rate"]["point"] == 0.0


def test_binary_report_has_no_group_block():
    report = evaluate(_AlwaysHate(), _toy_inputs(), n_bootstrap=20, write=False)
    assert report["target_groups"] == {}
