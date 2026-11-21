from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import TASK_THREE_CLASS
from src.groups import (
    agreement_table,
    annotator_agreement,
    available_groups,
    gradient_agreement,
    group_mask,
    human_hate_share,
    model_group_metrics,
    wilson,
)

TASK = TASK_THREE_CLASS
HATE, OFFENSIVE, NORMAL = 0, 1, 2


# Wilson interval


def test_wilson_matches_the_published_baseline():
    p = wilson(1393, 1804)
    assert round(100 * p.point, 1) == 77.2
    assert round(100 * p.lo, 1) == 75.2
    assert round(100 * p.hi, 1) == 79.1


def test_wilson_brackets_the_point_estimate():
    for k, n in [(1, 10), (50, 100), (99, 100), (0, 20), (20, 20)]:
        p = wilson(k, n)
        assert p.lo <= p.point <= p.hi


def test_wilson_stays_inside_zero_one():
    for k, n in [(0, 5), (5, 5), (1, 3)]:
        p = wilson(k, n)
        assert 0.0 <= p.lo and p.hi <= 1.0


def test_wilson_narrows_as_n_grows():
    small = wilson(50, 100)
    large = wilson(5000, 10000)
    assert (large.hi - large.lo) < (small.hi - small.lo)


def test_wilson_handles_an_empty_denominator():
    p = wilson(0, 0)
    assert p.point is None and p.lo is None
    assert not p.reliable


def test_reliability_bar_is_applied():
    assert not wilson(10, 50).reliable
    assert wilson(500, 1000).reliable


# group membership


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "post_id": [f"p{i}" for i in range(6)],
        "majority_label": ["hatespeech", "hatespeech", "offensive",
                           "offensive", "normal", "hatespeech"],
        "target_groups": [("islam",), ("islam",), ("islam",),
                          ("islam", "arab"), ("islam",), ("jewish",)],
        "annotator_labels": [
            ["hatespeech"] * 3,
            ["hatespeech", "hatespeech", "offensive"],
            ["offensive", "offensive", "hatespeech"],
            ["offensive", "offensive", "normal"],
            ["normal"] * 3,
            ["hatespeech"] * 3,
        ],
    })


def test_group_mask_matches_multi_target_posts(frame):
    assert group_mask(frame, "islam").sum() == 5
    assert group_mask(frame, "arab").sum() == 1
    assert group_mask(frame, "jewish").sum() == 1


def test_membership_is_not_exclusive(frame):
    assert group_mask(frame, "islam")[3]
    assert group_mask(frame, "arab")[3]


def test_available_groups_orders_by_hostile_count(frame):
    assert available_groups(frame)[0] == "islam"


# human hate-share


def test_human_hate_share_conditions_on_hostile_posts(frame):
    p = human_hate_share(frame, "islam")
    assert p.n == 4                      # 2 hate + 2 offensive, not the normal
    assert p.k == 2
    assert p.point == 0.5


def test_hate_share_is_none_for_an_absent_group(frame):
    assert human_hate_share(frame, "women").point is None


# model measures


def test_escalation_counts_offensive_called_hate(frame):
    # both offensive posts predicted hatespeech
    pred = np.array([HATE, HATE, HATE, HATE, NORMAL, HATE])
    m = model_group_metrics(frame, pred, "islam", TASK)
    assert m["escalation"].n == 2        # two human-offensive posts
    assert m["escalation"].point == 1.0


def test_de_escalation_counts_hate_called_offensive(frame):
    pred = np.array([OFFENSIVE, OFFENSIVE, OFFENSIVE, OFFENSIVE, NORMAL, HATE])
    m = model_group_metrics(frame, pred, "islam", TASK)
    assert m["de_escalation"].n == 2     # two human-hatespeech posts
    assert m["de_escalation"].point == 1.0


def test_escalation_and_de_escalation_are_not_mirrored(frame):
    pred = np.array([HATE, OFFENSIVE, HATE, OFFENSIVE, NORMAL, HATE])
    m = model_group_metrics(frame, pred, "islam", TASK)
    assert m["escalation"].point == 0.5      # 1 of 2 offensive escalated
    assert m["de_escalation"].point == 0.5   # 1 of 2 hate de-escalated
    assert m["escalation"].n == m["de_escalation"].n == 2


def test_model_hate_share_uses_the_models_own_hostile_set(frame):
    # model calls only one post hostile, and calls it hate
    pred = np.array([HATE, NORMAL, NORMAL, NORMAL, NORMAL, NORMAL])
    m = model_group_metrics(frame, pred, "islam", TASK)
    assert m["model_hate_share"].n == 1
    assert m["model_hate_share"].point == 1.0


def test_human_share_in_model_metrics_matches_the_standalone_measure(frame):
    pred = np.array([HATE] * 6)
    m = model_group_metrics(frame, pred, "islam", TASK)
    assert m["human_hate_share"].point == human_hate_share(frame, "islam").point


# annotator agreement


def test_unanimity_counts_only_full_agreement(frame):
    a = annotator_agreement(frame, "islam")
    assert a["n"] == 4                   # hostile posts only
    assert a["unanimity"] == 0.25        # only p0 is 3/3


def test_hate_offensive_split_isolates_severity_disagreement(frame):
    a = annotator_agreement(frame, "islam")
    assert a["hate_offensive_split"] == 0.5


def test_pairwise_agreement_is_between_zero_and_one(frame):
    a = annotator_agreement(frame, "islam")
    assert 0.0 <= a["pairwise_agreement"] <= 1.0


def test_agreement_table_covers_every_requested_group(frame):
    tbl = agreement_table(frame, ("islam", "jewish", "women"))
    assert list(tbl["group"]) == ["islam", "jewish", "women"]
    assert tbl.loc[tbl["group"] == "women", "n_hostile"].iloc[0] == 0


# gradient agreement


def _gradient_frame(human, model, n=1000) -> pd.DataFrame:
    return pd.DataFrame({
        "group": [f"g{i}" for i in range(len(human))],
        "n_human_hostile": [n] * len(human),
        "human_hate_share": human,
        "model_hate_share": model,
    })


def test_perfect_ordering_gives_rho_one():
    a = gradient_agreement(_gradient_frame([0.8, 0.6, 0.4], [0.9, 0.7, 0.5]))
    assert a["spearman_rho"] == pytest.approx(1.0)


def test_reversed_ordering_gives_rho_minus_one():
    a = gradient_agreement(_gradient_frame([0.8, 0.6, 0.4], [0.5, 0.7, 0.9]))
    assert a["spearman_rho"] == pytest.approx(-1.0)


def test_small_n_uses_an_exact_permutation_p_value():
    a = gradient_agreement(_gradient_frame([0.8, 0.7, 0.6, 0.5, 0.4],
                                           [0.9, 0.8, 0.7, 0.6, 0.5]))
    assert "permutation" in a["p_method"]
    # smallest attainable two-sided p at n=5 is 2/120
    assert a["p_value"] == pytest.approx(2 / 120)
    assert a["p_value"] > 0.0


def test_mean_shift_reports_level_not_just_ordering():
    a = gradient_agreement(_gradient_frame([0.5, 0.4, 0.3], [0.7, 0.6, 0.5]))
    assert a["spearman_rho"] == pytest.approx(1.0)
    assert a["mean_shift"] == pytest.approx(0.2)


def test_groups_below_the_reliability_bar_are_excluded():
    df = _gradient_frame([0.8, 0.6, 0.4, 0.2], [0.9, 0.7, 0.5, 0.3])
    df.loc[0, "n_human_hostile"] = 10
    a = gradient_agreement(df, min_n=200)
    assert a["n_groups"] == 3
    assert "g0" not in a["groups"]


def test_too_few_groups_returns_no_correlation():
    a = gradient_agreement(_gradient_frame([0.8, 0.6], [0.9, 0.7]))
    assert a["spearman_rho"] is None
