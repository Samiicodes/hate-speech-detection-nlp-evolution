from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# directory -> the task every report in it must declare.
TASK_BY_DIR: dict[str, str] = {
    "results": "three_class",
    "results_binary": "hate_only",
}

# Metrics that only mean something under a single positive class.
BINARY_ONLY_METRICS = ("over_flag_rate", "under_detect_rate")

# Derived analyses, no task block
DERIVED_THREE_CLASS = (
    "label_correction_*.json",
    "dose_response.json",
    "corrected_feature_analysis.json",
)


def _model_reports(directory: str) -> list[tuple[Path, dict]]:
    out = []
    for path in sorted((PROJECT_ROOT / directory).glob("*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(blob.get("task"), dict) and "headline" in blob:
            out.append((path, blob))
    return out


def _all_model_reports() -> list[tuple[str, Path, dict]]:
    return [(d, p, b) for d in TASK_BY_DIR for p, b in _model_reports(d)]


def declared_n_classes(task: dict) -> int:
    if "n_classes" in task:
        return int(task["n_classes"])
    for field in ("class_names", "label_map"):
        if field in task:
            return len(task[field])
    if "positive_labels" in task:
        return 2
    raise AssertionError(
        f"task block {task.get('name')!r} does not say how many classes it "
        "has by any route -- a reader cannot tell what its metrics mean"
    )


def test_every_results_directory_has_model_reports():
    for directory in TASK_BY_DIR:
        assert _model_reports(directory), f"no model reports found in {directory}/"


@pytest.mark.parametrize("directory", sorted(TASK_BY_DIR))
def test_declared_task_matches_the_directory(directory):
    expected = TASK_BY_DIR[directory]
    for path, blob in _model_reports(directory):
        assert blob["task"]["name"] == expected, (
            f"{path.name} lives in {directory}/ but declares task "
            f"{blob['task']['name']!r}, not {expected!r}. A file in the wrong "
            f"directory is how a binary number gets quoted as a three-class one."
        )


def test_declared_n_classes_matches_the_confusion_matrix():
    for directory, path, blob in _all_model_reports():
        n = declared_n_classes(blob["task"])
        cm = blob["headline"]["confusion_matrix"]
        assert len(cm["labels"]) == n, f"{directory}/{path.name}: labels != n_classes"
        assert len(cm["matrix"]) == n, f"{directory}/{path.name}: matrix rows != n_classes"
        assert all(len(row) == n for row in cm["matrix"]), (
            f"{directory}/{path.name}: confusion matrix is not {n}x{n}"
        )
        assert len(blob["headline"]["per_class"]) == n, (
            f"{directory}/{path.name}: per_class has "
            f"{len(blob['headline']['per_class'])} entries, task declares {n}"
        )


def test_binary_only_metrics_appear_only_under_the_binary_task():
    for directory, path, blob in _all_model_reports():
        binary = declared_n_classes(blob["task"]) == 2
        for metric in BINARY_ONLY_METRICS:
            present = metric in blob["headline"]
            if binary:
                assert present, (
                    f"{directory}/{path.name} is binary but has no {metric}"
                )
            else:
                assert not present, (
                    f"{directory}/{path.name} declares "
                    f"{blob['task']['name']!r} yet reports {metric!r}, which is "
                    "a single-positive-class notion. Under three classes this "
                    "quantity splits into escalation and de-escalation."
                )


def test_binary_only_metrics_absent_from_slices_too():
    for directory, path, blob in _all_model_reports():
        if declared_n_classes(blob["task"]) == 2:
            continue
        for name, slice_blob in blob.get("slices", {}).items():
            for metric in BINARY_ONLY_METRICS:
                assert metric not in slice_blob, (
                    f"{directory}/{path.name}: slice {name!r} reports {metric!r} "
                    f"under {blob['task']['name']!r}"
                )


def test_hatecheck_mappings_are_named_not_implied():
    for directory, path, blob in _all_model_reports():
        hc = blob.get("hatecheck_muslim")
        if hc is None:
            continue
        if declared_n_classes(blob["task"]) == 2:
            # Binary gold, one mapping only
            assert "lenient" not in hc, (
                f"{directory}/{path.name} is binary but carries a 'lenient' "
                "HateCheck block. Strict and lenient are three-class notions; "
                "a binary file offering both invites the reader to pick one."
            )
            continue
        assert "mapping" in hc, (
            f"{directory}/{path.name}: three-class HateCheck block does not "
            "name its mapping, and the strict and lenient rates differ by "
            "roughly a factor of ten"
        )
        assert "strict" in hc["mapping"], (
            f"{directory}/{path.name}: top-level HateCheck mapping should be "
            "the strict one under three classes"
        )
        assert "lenient" in hc and "mapping" in hc["lenient"], (
            f"{directory}/{path.name}: three-class HateCheck must also carry "
            "the lenient mapping, named"
        )


def test_derived_analyses_are_three_class_by_construction():
    task = config.DEFAULT_TASK
    assert task.n_classes == 3, "this suite assumes the three-class default"

    found = 0
    for pattern in DERIVED_THREE_CLASS:
        for path in sorted((PROJECT_ROOT / "results").glob(pattern)):
            found += 1
            blob = json.loads(path.read_text(encoding="utf-8"))
            correction = blob.get("correction")
            if correction is not None:
                share = correction["target_hate_share"]
                assert 0.0 < share < 1.0, f"{path.name}: implausible target share"
                assert correction["n_reweighted"] < correction["n_posts"], (
                    f"{path.name}: every post reweighted -- that is what a "
                    "binary corpus would look like here, with no offensive "
                    "class to re-level against"
                )
            gradient = blob.get("gradient")
            if gradient is not None:
                groups = gradient["baseline"]["groups"]
                assert groups == list(config.ANALYSIS_GROUPS), (
                    f"{path.name}: gradient groups do not match "
                    "config.ANALYSIS_GROUPS"
                )
    assert found, "no derived three-class analyses found to check"


def test_out_of_fold_predictions_are_three_class_and_full_corpus():
    task = config.DEFAULT_TASK
    paths = sorted((PROJECT_ROOT / "results" / "crossval").glob("*_oof.json"))
    assert paths, "no out-of-fold prediction files found"
    for path in paths:
        blob = json.loads(path.read_text(encoding="utf-8"))
        assert blob["task"]["name"] == task.name, (
            f"{path.name} declares {blob['task']['name']!r} but the group "
            f"analysis assumes {task.name!r}"
        )
        assert declared_n_classes(blob["task"]) == task.n_classes
        for seed, preds in blob["predictions"].items():
            observed = set(preds)
            assert observed <= set(range(task.n_classes)), (
                f"{path.name} seed {seed}: predictions contain "
                f"{sorted(observed - set(range(task.n_classes)))}, outside the "
                f"declared {task.n_classes}-class label space"
            )


def test_binary_and_three_class_headline_numbers_actually_differ():
    three = {p.stem: b for p, b in _model_reports("results")}
    binary = {p.stem: b for p, b in _model_reports("results_binary")}
    shared = sorted(set(three) & set(binary))
    assert shared, "no model appears under both tasks"
    for name in shared:
        a = three[name]["headline"]["macro_f1"]
        b = binary[name]["headline"]["macro_f1"]
        assert abs(a - b) > 1e-6, (
            f"{name}: three-class and binary macro-F1 are identical ({a}). "
            "One directory has been overwritten with the other task's results."
        )
