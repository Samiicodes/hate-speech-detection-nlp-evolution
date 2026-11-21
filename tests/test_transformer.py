from __future__ import annotations

import numpy as np
import pytest

from src.harness import Detector, aggregate_reports
from src.models import transformer as tf_mod

torch = pytest.importorskip("torch", reason="system 4 needs torch")


# seeding


def test_set_all_seeds_makes_torch_reproducible():
    tf_mod.set_all_seeds(123)
    a = torch.randn(8)
    tf_mod.set_all_seeds(123)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_set_all_seeds_also_seeds_numpy_and_random():
    import random

    tf_mod.set_all_seeds(7)
    first = (np.random.rand(3).tolist(), random.random())
    tf_mod.set_all_seeds(7)
    assert (np.random.rand(3).tolist(), random.random()) == first


def test_different_seeds_give_different_draws():
    tf_mod.set_all_seeds(1)
    a = torch.randn(8)
    tf_mod.set_all_seeds(2)
    assert not torch.equal(a, torch.randn(8))


# the Detector contract


def test_detector_satisfies_the_protocol():
    assert isinstance(tf_mod.TransformerDetector(), Detector)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        tf_mod.TransformerDetector().predict(["anything"])


def test_params_record_the_training_recipe():
    det = tf_mod.TransformerDetector(seed=99, epochs=5, lr=3e-5)
    params = det.params()
    assert params["seed"] == 99
    assert params["epochs"] == 5
    assert params["lr"] == 3e-5
    assert params["class_weighted"] is True


# aggregation


def _fake_report(seed: int, acc: float, gap: float) -> dict:
    return {
        "model_name": f"m_seed{seed}",
        "seed": seed,
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
        "task": {"name": "hate_only"},
        "split": {"seed": 42},
        "environment": {"python": "3.13"},
        "params": {"seed": seed},
        "notes": "",
        "headline": {"accuracy": acc, "macro_f1": acc - 0.02, "n": 100},
        "key_metrics": {
            "over_flag_gap": gap,
            "macro_f1_gap": -gap,
            "over_flag_islam": 0.2, "over_flag_rest": 0.1,
            "macro_f1_islam": 0.7, "macro_f1_rest": 0.8,
            "hatecheck_key_functionalities": {"ident_pos_nh": acc},
        },
        "hatecheck_muslim": {
            "recall_on_hateful": acc, "false_alarm_on_non_hateful": gap,
        },
        "cost": {"train_seconds": 10.0 * seed},
        "predictions": {"post_ids": ["a"], "y_true": [1], "y_pred": [seed % 2]},
        "hatecheck_predictions": {"case_ids": ["1"], "y_true": [1], "y_pred": [1]},
    }


def test_aggregate_averages_numeric_leaves():
    reports = [_fake_report(1, 0.80, 0.10), _fake_report(2, 0.90, 0.20)]
    agg = aggregate_reports(reports, "m", write=False)
    assert agg["headline"]["accuracy"] == pytest.approx(0.85)
    assert agg["key_metrics"]["over_flag_gap"] == pytest.approx(0.15)


def test_aggregate_records_sample_std():
    reports = [_fake_report(1, 0.80, 0.10), _fake_report(2, 0.90, 0.20)]
    agg = aggregate_reports(reports, "m", write=False)
    expected = float(np.std([0.80, 0.90], ddof=1))
    assert agg["runs"]["std"]["headline"]["accuracy"] == pytest.approx(expected)


def test_aggregate_does_not_average_predictions():
    reports = [_fake_report(1, 0.8, 0.1), _fake_report(2, 0.9, 0.2)]
    agg = aggregate_reports(reports, "m", write=False)
    assert agg["predictions"] == reports[0]["predictions"]
    assert agg["runs"]["predictions_from_seed"] == 1


def test_aggregate_keeps_one_row_per_seed():
    reports = [_fake_report(s, 0.8, 0.1) for s in (1, 2, 3)]
    agg = aggregate_reports(reports, "m", write=False)
    assert agg["runs"]["n_seeds"] == 3
    assert agg["runs"]["seeds"] == [1, 2, 3]
    assert len(agg["runs"]["per_seed"]) == 3


def test_aggregate_of_a_single_run_has_zero_std():
    agg = aggregate_reports([_fake_report(1, 0.8, 0.1)], "m", write=False)
    assert agg["runs"]["std"]["headline"]["accuracy"] == 0.0


def test_aggregate_rejects_an_empty_list():
    with pytest.raises(ValueError):
        aggregate_reports([], "m", write=False)


def test_aggregate_is_json_serialisable():
    import json

    json.dumps(aggregate_reports([_fake_report(1, 0.8, 0.1)], "m", write=False))


# a real (tiny) fine-tune


@pytest.mark.slow
def test_two_step_finetune_runs_end_to_end(dataset):
    train, _, _ = dataset
    det = tf_mod.TransformerDetector(
        model_name="distilroberta-base", epochs=1, batch_size=8, max_length=64,
    )
    det.fit(train["text"].tolist()[:16], train["label"].tolist()[:16])
    preds = det.predict(["muslims are scum", "have a nice day"])
    assert len(preds) == 2
    assert set(preds) <= {0, 1}
