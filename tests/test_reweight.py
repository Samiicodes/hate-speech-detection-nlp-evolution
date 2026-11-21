from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.reweight import corpus_hate_share, group_balance_weights, verify_correction

GROUPS = ("alpha", "beta")


def _frame(alpha_hate: int, alpha_off: int, beta_hate: int, beta_off: int,
           overlap: int = 0, n_normal: int = 0) -> pd.DataFrame:
    rows = []
    for _ in range(alpha_hate):
        rows.append(("hatespeech", ("alpha",)))
    for _ in range(alpha_off):
        rows.append(("offensive", ("alpha",)))
    for _ in range(beta_hate):
        rows.append(("hatespeech", ("beta",)))
    for _ in range(beta_off):
        rows.append(("offensive", ("beta",)))
    for i in range(overlap):
        rows.append(("hatespeech" if i % 2 else "offensive", ("alpha", "beta")))
    for _ in range(n_normal):
        rows.append(("normal", ()))
    return pd.DataFrame({
        "post_id": [f"p{i}" for i in range(len(rows))],
        "majority_label": [r[0] for r in rows],
        "target_groups": [r[1] for r in rows],
        "label": [{"hatespeech": 0, "offensive": 1, "normal": 2}[r[0]] for r in rows],
    })


# the target


def test_corpus_hate_share_ignores_normal_posts():
    df = _frame(60, 40, 0, 0, n_normal=1000)
    assert corpus_hate_share(df) == pytest.approx(0.6)


def test_default_target_is_the_corpus_share():
    df = _frame(80, 20, 20, 80)
    weights, report = group_balance_weights(df, GROUPS)
    assert report["target_hate_share"] == pytest.approx(corpus_hate_share(df))


# exactness


def test_disjoint_groups_are_equalised_exactly():
    df = _frame(80, 20, 20, 80)
    weights, report = group_balance_weights(df, GROUPS, target_share=0.5)
    check = verify_correction(df, weights, GROUPS)
    for value in check["share_after_weighted"]:
        assert value == pytest.approx(0.5, abs=1e-9)


def test_overlapping_groups_are_equalised_exactly():
    df = _frame(80, 20, 20, 80, overlap=60)
    weights, report = group_balance_weights(df, GROUPS, target_share=0.5)
    check = verify_correction(df, weights, GROUPS)
    for value in check["share_after_weighted"]:
        assert value == pytest.approx(0.5, abs=1e-8), (
            "overlapping groups not equalised -- residual gradient would "
            "contaminate the causal test"
        )
    assert report["ipf_converged"]
    assert report["ipf_max_deviation"] < 1e-8


def test_report_records_convergence():
    df = _frame(80, 20, 20, 80, overlap=40)
    _, report = group_balance_weights(df, GROUPS, target_share=0.5)
    assert report["ipf_iterations"] >= 1
    assert "iterative proportional fitting" in report["method"]


def test_an_arbitrary_target_is_hit():
    df = _frame(90, 10, 10, 90, overlap=20)
    weights, _ = group_balance_weights(df, GROUPS, target_share=0.25)
    check = verify_correction(df, weights, GROUPS)
    for value in check["share_after_weighted"]:
        assert value == pytest.approx(0.25, abs=1e-8)


# what must not change


def test_normal_posts_keep_weight_one():
    df = _frame(80, 20, 20, 80, n_normal=100)
    weights, _ = group_balance_weights(df, GROUPS, target_share=0.5,
                                       normalise=False)
    normal = (df["majority_label"] == "normal").to_numpy()
    assert np.allclose(weights[normal], 1.0)


def test_posts_outside_the_analysis_groups_are_untouched():
    df = _frame(80, 20, 20, 80)
    df.loc[0, "target_groups"] = ("gamma",)      # not in GROUPS
    weights, report = group_balance_weights(df, GROUPS, target_share=0.5,
                                            normalise=False)
    assert weights[0] == pytest.approx(1.0)
    assert report["n_untouched"] >= 1


def test_no_post_is_dropped():
    df = _frame(80, 20, 20, 80, overlap=30, n_normal=50)
    weights, _ = group_balance_weights(df, GROUPS)
    assert len(weights) == len(df)
    assert (weights > 0).all(), "a zero weight would silently drop a post"


def test_weights_are_normalised_to_mean_one_by_default():
    df = _frame(80, 20, 20, 80)
    weights, _ = group_balance_weights(df, GROUPS)
    assert weights.mean() == pytest.approx(1.0)


def test_already_balanced_groups_are_barely_touched():
    df = _frame(50, 50, 50, 50)
    weights, _ = group_balance_weights(df, GROUPS, target_share=0.5,
                                       normalise=False)
    assert np.allclose(weights, 1.0)


# cost reporting


def test_effective_sample_size_is_reported_and_below_n():
    df = _frame(95, 5, 5, 95)
    _, report = group_balance_weights(df, GROUPS, target_share=0.5)
    assert report["effective_sample_size"] < report["n_posts"]
    assert 0 < report["ess_fraction"] <= 1


def test_a_severe_correction_costs_more_effective_sample():
    mild, _ = group_balance_weights(_frame(55, 45, 45, 55), GROUPS,
                                    target_share=0.5)
    severe_df = _frame(95, 5, 5, 95)
    _, severe_report = group_balance_weights(severe_df, GROUPS,
                                             target_share=0.5)
    _, mild_report = group_balance_weights(_frame(55, 45, 45, 55), GROUPS,
                                           target_share=0.5)
    assert severe_report["ess_fraction"] < mild_report["ess_fraction"]


def test_degenerate_group_is_skipped_not_crashed():
    df = _frame(50, 50, 30, 0)      # beta has no offensive posts
    weights, report = group_balance_weights(df, GROUPS, target_share=0.5)
    assert "note" in report["per_group"]["beta"]
    assert np.isfinite(weights).all()
