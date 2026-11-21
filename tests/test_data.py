from __future__ import annotations

import pytest

from src import config, data
from src.config import TASK_BINARY, TASK_HATE_OR_OFFENSIVE, TASK_THREE_CLASS


# majority vote


@pytest.mark.parametrize(
    "labels,expected",
    [
        (["hatespeech", "hatespeech", "normal"], "hatespeech"),
        (["normal", "normal", "normal"], "normal"),
        (["offensive", "normal", "offensive"], "offensive"),
        (["hatespeech", "offensive", "normal"], data.TIED),
        ([], data.TIED),
    ],
)
def test_majority_label(labels, expected):
    assert data.majority_label(labels) == expected


def test_target_votes_dedupe_within_annotator():
    annotators = [
        {"target": ["Islam", "Islam"]},
        {"target": ["Islam", "Jewish"]},
        {"target": ["None"]},
    ]
    counts = data.target_vote_counts(annotators)
    assert counts["islam"] == 2
    assert counts["jewish"] == 1


# pilot counts


def test_pilot_counts_hatexplain(hatexplain_all):
    observed = data.hatexplain_summary(hatexplain_all)
    for key, expected in observed.items():
        assert observed[key] == config.EXPECTED[key], (
            f"{key}: expected {config.EXPECTED[key]}, got {observed[key]}"
        )


def test_pilot_counts_hatecheck(hatecheck):
    observed = data.hatecheck_summary(hatecheck)
    for key, expected in observed.items():
        assert observed[key] == config.EXPECTED[key]


def test_ties_dropped_leaves_19229(hatexplain_labelled):
    expected = config.EXPECTED["hatexplain_total"] - config.EXPECTED["majority_tied"]
    assert len(hatexplain_labelled) == expected
    assert not hatexplain_labelled["is_tied"].any()


def test_islam_slice_size_after_dropping_ties(hatexplain_labelled):
    n_slice = int(hatexplain_labelled["is_islam_slice"].sum())
    assert n_slice == (
        config.EXPECTED["islam_slice_total"] - config.EXPECTED["islam_slice_tied"]
    )


# binary task definition


def test_default_task_is_three_class():
    assert config.DEFAULT_TASK.name == "three_class"
    assert config.DEFAULT_TASK.n_classes == 3
    assert not config.DEFAULT_TASK.is_binary


def test_three_class_mapping_is_severity_ordered():
    assert TASK_THREE_CLASS.to_label("hatespeech") == 0
    assert TASK_THREE_CLASS.to_label("offensive") == 1
    assert TASK_THREE_CLASS.to_label("normal") == 2
    assert TASK_THREE_CLASS.class_names == ("hatespeech", "offensive", "normal")
    assert TASK_THREE_CLASS.index_of("hatespeech") == 0


def test_archived_binary_task_still_maps_as_before():
    assert TASK_BINARY.to_label("hatespeech") == 1
    assert TASK_BINARY.to_label("offensive") == 0
    assert TASK_BINARY.to_label("normal") == 0
    assert TASK_BINARY.is_binary


def test_alternative_task_folds_offensive_in():
    assert TASK_HATE_OR_OFFENSIVE.to_label("offensive") == 1
    assert TASK_HATE_OR_OFFENSIVE.to_label("normal") == 0


def test_three_class_counts_match_the_pilot(hatexplain_labelled):
    counts = hatexplain_labelled["label"].value_counts()
    assert int(counts[0]) == config.EXPECTED["majority_hatespeech"]
    assert int(counts[1]) == config.EXPECTED["majority_offensive"]
    assert int(counts[2]) == config.EXPECTED["majority_normal"]


def test_binary_labels_are_recoverable_from_the_same_rows(hatexplain_labelled):
    binary = data.labelled_hatexplain(task=TASK_BINARY)
    assert binary["post_id"].tolist() == hatexplain_labelled["post_id"].tolist()
    assert int(binary["label"].sum()) == config.EXPECTED["majority_hatespeech"]


# splits


def test_splits_are_disjoint_and_complete(splits, hatexplain_labelled):
    train, val, test = set(splits.train), set(splits.val), set(splits.test)
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(hatexplain_labelled["post_id"])


def test_splits_are_deterministic(hatexplain_labelled):
    a = data.make_splits(hatexplain_labelled, config.DEFAULT_SPLIT)
    b = data.make_splits(hatexplain_labelled, config.DEFAULT_SPLIT)
    assert a.train == b.train and a.val == b.val and a.test == b.test


def test_split_proportions(splits, hatexplain_labelled):
    n = len(hatexplain_labelled)
    assert len(splits.train) / n == pytest.approx(0.70, abs=0.01)
    assert len(splits.val) / n == pytest.approx(0.15, abs=0.01)
    assert len(splits.test) / n == pytest.approx(0.15, abs=0.01)


def test_stratification_holds_for_label_and_slice(dataset, hatexplain_labelled):
    overall_slice = hatexplain_labelled["is_islam_slice"].mean()
    for cls in range(config.DEFAULT_TASK.n_classes):
        overall = (hatexplain_labelled["label"] == cls).mean()
        for part in dataset:
            assert (part["label"] == cls).mean() == pytest.approx(overall, abs=0.015)
    for part in dataset:
        assert part["is_islam_slice"].mean() == pytest.approx(overall_slice, abs=0.01)


def test_saved_splits_are_reused_not_rebuilt(splits, hatexplain_labelled):
    fresh = data.make_splits(hatexplain_labelled, config.DEFAULT_SPLIT)
    assert fresh.test != splits.test, (
        "a fresh three-class build now matches the saved split -- if that is "
        "intended, the comparability argument in the README needs revisiting"
    )
    # still a valid partition
    assert len(splits.train) + len(splits.val) + len(splits.test) == len(
        hatexplain_labelled
    )


def test_split_rebuild_is_deterministic_under_the_current_task(hatexplain_labelled):
    a = data.make_splits(hatexplain_labelled, config.DEFAULT_SPLIT)
    b = data.make_splits(hatexplain_labelled, config.DEFAULT_SPLIT)
    assert a.test == b.test


def test_hatecheck_is_never_in_the_training_data(dataset, hatecheck):
    train_texts = set(dataset[0]["text"].str.lower())
    hc_texts = set(hatecheck["test_case"].str.lower().str.strip())
    assert not (train_texts & hc_texts)
