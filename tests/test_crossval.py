from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import crossval
from src.harness import BaseDetector


@pytest.fixture
def toy_corpus() -> pd.DataFrame:
    n = 200
    return pd.DataFrame({
        "post_id": [f"p{i:04d}" for i in range(n)],
        "text": [f"text number {i}" for i in range(n)],
        "label": [i % 3 for i in range(n)],
        "is_islam_slice": [i % 5 == 0 for i in range(n)],
        "majority_label": [
            ["hatespeech", "offensive", "normal"][i % 3] for i in range(n)
        ],
        "target_groups": [("islam",) if i % 5 == 0 else ("jewish",)
                          for i in range(n)],
    })


# folds


def test_folds_cover_every_post_exactly_once(toy_corpus):
    folds = crossval.make_folds(toy_corpus, n_splits=5)
    assert len(folds.post_ids) == len(toy_corpus)
    assert set(folds.post_ids) == set(toy_corpus["post_id"])
    assert sorted(set(folds.fold)) == [0, 1, 2, 3, 4]


def test_folds_are_roughly_equal_sized(toy_corpus):
    folds = crossval.make_folds(toy_corpus, n_splits=5)
    counts = pd.Series(folds.fold).value_counts()
    assert counts.max() - counts.min() <= 1


def test_folds_are_deterministic(toy_corpus):
    a = crossval.make_folds(toy_corpus, n_splits=5, seed=42)
    b = crossval.make_folds(toy_corpus, n_splits=5, seed=42)
    assert a.fold == b.fold


def test_folds_change_with_the_seed(toy_corpus):
    a = crossval.make_folds(toy_corpus, n_splits=5, seed=1)
    b = crossval.make_folds(toy_corpus, n_splits=5, seed=2)
    assert a.fold != b.fold


def test_folds_preserve_class_balance(toy_corpus):
    folds = crossval.make_folds(toy_corpus, n_splits=5)
    df = toy_corpus.sort_values("post_id").reset_index(drop=True)
    df["fold"] = folds.fold
    overall = df["label"].value_counts(normalize=True)
    for k in range(5):
        part = df[df["fold"] == k]["label"].value_counts(normalize=True)
        for cls in overall.index:
            assert part.get(cls, 0) == pytest.approx(overall[cls], abs=0.08)


def test_folds_round_trip(toy_corpus, tmp_path):
    folds = crossval.make_folds(toy_corpus, n_splits=5)
    path = crossval.save_folds(folds, tmp_path / "f.json")
    back = crossval.load_folds(path)
    assert back.fold == folds.fold
    assert back.stratify == folds.stratify


# out-of-fold prediction


class _RecordingDetector(BaseDetector):

    def __init__(self, log: dict) -> None:
        self.name = "recording"
        self.log = log
        self.seen_fit: list[str] = []
        self.seen_tune: list[str] = []

    def fit(self, texts, labels):
        self.seen_fit = list(texts)
        self.log["fit"].append(set(texts))

    def tune(self, texts, labels):
        self.seen_tune = list(texts)
        self.log["tune"].append(set(texts))

    def predict(self, texts):
        self.log["predict"].append(set(texts))
        return [0] * len(texts)


def test_no_post_is_predicted_by_a_model_that_trained_on_it(toy_corpus):
    folds = crossval.make_folds(toy_corpus, n_splits=5)
    log = {"fit": [], "tune": [], "predict": []}
    crossval.out_of_fold_predictions(
        toy_corpus.sort_values("post_id").reset_index(drop=True),
        folds,
        lambda fold, seed: _RecordingDetector(log),
        progress=lambda _: None,
    )
    assert len(log["predict"]) == 5
    for fitted, predicted in zip(log["fit"], log["predict"]):
        assert not (fitted & predicted), "a fold was predicted by a model that saw it"


def test_tuning_never_touches_the_predicted_fold(toy_corpus):
    folds = crossval.make_folds(toy_corpus, n_splits=5)
    log = {"fit": [], "tune": [], "predict": []}
    crossval.out_of_fold_predictions(
        toy_corpus.sort_values("post_id").reset_index(drop=True),
        folds,
        lambda fold, seed: _RecordingDetector(log),
        progress=lambda _: None,
    )
    for tuned, predicted in zip(log["tune"], log["predict"]):
        assert not (tuned & predicted), "tuning saw the fold being predicted"


def test_inner_split_is_disjoint(toy_corpus):
    inner_train, inner_val = crossval.inner_split(toy_corpus, seed=42)
    assert not set(inner_train["post_id"]) & set(inner_val["post_id"])
    assert len(inner_train) + len(inner_val) == len(toy_corpus)


def test_every_post_gets_a_prediction(toy_corpus):
    folds = crossval.make_folds(toy_corpus, n_splits=5)
    ordered = toy_corpus.sort_values("post_id").reset_index(drop=True)

    class _One(BaseDetector):
        name = "one"

        def fit(self, texts, labels):
            pass

        def predict(self, texts):
            return [1] * len(texts)

    preds = crossval.out_of_fold_predictions(
        ordered, folds, lambda f, s: _One(), progress=lambda _: None
    )
    assert len(preds) == len(ordered)
    assert (preds == 1).all()


def test_missing_fold_assignment_is_an_error(toy_corpus):
    folds = crossval.make_folds(toy_corpus.iloc[:100], n_splits=5)
    with pytest.raises(ValueError, match="no fold assignment"):
        crossval.out_of_fold_predictions(
            toy_corpus.sort_values("post_id").reset_index(drop=True),
            folds,
            lambda f, s: _RecordingDetector({"fit": [], "tune": [],
                                             "predict": []}),
            progress=lambda _: None,
        )


def test_a_fresh_detector_is_built_per_fold(toy_corpus):
    folds = crossval.make_folds(toy_corpus, n_splits=3)
    built = []

    def _build(fold, seed):
        built.append(fold)
        return _RecordingDetector({"fit": [], "tune": [], "predict": []})

    crossval.out_of_fold_predictions(
        toy_corpus.sort_values("post_id").reset_index(drop=True),
        folds, _build, progress=lambda _: None,
    )
    assert built == [0, 1, 2]


# storage


def test_sample_weights_reach_the_detector(toy_corpus):
    folds = crossval.make_folds(toy_corpus, n_splits=5)
    ordered = toy_corpus.sort_values("post_id").reset_index(drop=True)
    seen: list[np.ndarray] = []

    class _WeightAware(BaseDetector):
        name = "weight_aware"
        supports_sample_weight = True

        def fit(self, texts, labels, sample_weight=None):
            seen.append(np.asarray(sample_weight))

        def predict(self, texts):
            return [0] * len(texts)

    weights = np.linspace(0.5, 2.0, len(ordered))
    crossval.out_of_fold_predictions(
        ordered, folds, lambda f, s: _WeightAware(),
        progress=lambda _: None, sample_weight=weights,
    )
    assert len(seen) == 5
    for w in seen:
        assert w is not None and len(w) > 0
        assert w.min() >= weights.min() and w.max() <= weights.max()


def test_weights_on_an_unsupporting_detector_raise(toy_corpus):
    folds = crossval.make_folds(toy_corpus, n_splits=5)
    ordered = toy_corpus.sort_values("post_id").reset_index(drop=True)

    class _Plain(BaseDetector):
        name = "plain"

        def fit(self, texts, labels):
            pass

        def predict(self, texts):
            return [0] * len(texts)

    with pytest.raises(TypeError, match="does not support sample_weight"):
        crossval.out_of_fold_predictions(
            ordered, folds, lambda f, s: _Plain(),
            progress=lambda _: None,
            sample_weight=np.ones(len(ordered)),
        )


def test_mismatched_weight_length_is_an_error(toy_corpus):
    folds = crossval.make_folds(toy_corpus, n_splits=5)
    ordered = toy_corpus.sort_values("post_id").reset_index(drop=True)
    with pytest.raises(ValueError, match="sample_weight has"):
        crossval.out_of_fold_predictions(
            ordered, folds, lambda f, s: None,
            progress=lambda _: None, sample_weight=np.ones(5),
        )


def test_oof_round_trip(toy_corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(crossval, "CV_DIR", tmp_path)
    folds = crossval.make_folds(toy_corpus, n_splits=5)
    ordered = toy_corpus.sort_values("post_id").reset_index(drop=True)
    preds = {42: [0] * len(ordered), 43: [1] * len(ordered)}
    crossval.save_oof("toy", ordered, preds, folds)

    blob = crossval.load_oof("toy")
    assert blob["seeds"] == [42, 43]
    assert blob["n_posts"] == len(ordered)
    assert blob["post_ids"] == ordered["post_id"].tolist()
    assert blob["predictions"]["42"] == [0] * len(ordered)
