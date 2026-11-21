from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.config import DEFAULT_TASK
from src.harness import Detector
from src.models.naive_bayes import NaiveBayesDetector


@pytest.fixture(scope="module")
def fitted(dataset):
    train, val, _ = dataset
    det = NaiveBayesDetector()
    det.fit(train["text"].tolist(), train["label"].tolist())
    det.tune(val["text"].tolist(), val["label"].tolist())
    return det


# the configuration choices


def test_uses_raw_counts_not_tfidf(fitted):
    vec = fitted.pipeline.named_steps["counts"]
    assert isinstance(vec, CountVectorizer)
    assert not isinstance(vec, TfidfVectorizer)


def test_headline_configuration_is_unigrams(fitted):
    assert fitted.ngram_range == (1, 1)
    names = fitted.pipeline.named_steps["counts"].get_feature_names_out()
    assert not any(" " in str(n) for n in names)


def test_class_prior_is_uniform_by_default(fitted):
    assert fitted.fit_prior is False
    prior = np.exp(fitted.pipeline.named_steps["clf"].class_log_prior_)
    assert np.allclose(prior, 1 / DEFAULT_TASK.n_classes)
    assert fitted.params()["class_prior"] == "uniform"


def test_empirical_prior_is_available_as_a_sensitivity_check(dataset):
    train, _, _ = dataset
    det = NaiveBayesDetector(fit_prior=True, tune_grid=None)
    det.fit(train["text"].tolist()[:3000], train["label"].tolist()[:3000])
    prior = np.exp(det.pipeline.named_steps["clf"].class_log_prior_)
    assert not np.allclose(prior, 1 / 3)
    assert det.params()["class_prior"] == "empirical"


def test_only_smoothing_is_tuned(fitted):
    assert fitted.tuning_history
    assert {row["alpha"] for row in fitted.tuning_history} == set(fitted.tune_grid)
    for row in fitted.tuning_history:
        assert set(row) == {"alpha", "val_macro_f1"}


# the Detector contract


def test_satisfies_the_protocol():
    assert isinstance(NaiveBayesDetector(), Detector)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        NaiveBayesDetector().predict(["anything"])


def test_predict_returns_one_class_per_text(fitted):
    preds = fitted.predict(["muslims are scum", "have a nice day"])
    assert len(preds) == 2
    assert set(preds) <= set(range(DEFAULT_TASK.n_classes))


def test_is_deterministic(dataset):
    train, _, test = dataset
    texts, labels = train["text"].tolist()[:2000], train["label"].tolist()[:2000]
    a, b = NaiveBayesDetector(tune_grid=None), NaiveBayesDetector(tune_grid=None)
    a.fit(texts, labels)
    b.fit(texts, labels)
    sample = test["text"].tolist()[:200]
    assert a.predict(sample) == b.predict(sample)


# sample weights, for the causal


def test_advertises_sample_weight_support():
    assert NaiveBayesDetector.supports_sample_weight is True


def test_sample_weights_change_the_fit(dataset):
    train, _, _ = dataset
    texts, labels = train["text"].tolist()[:3000], train["label"].tolist()[:3000]

    plain = NaiveBayesDetector(tune_grid=None)
    plain.fit(texts, labels)

    rng = np.random.default_rng(0)
    weighted = NaiveBayesDetector(tune_grid=None)
    weighted.fit(texts, labels, sample_weight=rng.uniform(0.1, 3.0, len(texts)))

    a = plain.pipeline.named_steps["clf"].feature_log_prob_
    b = weighted.pipeline.named_steps["clf"].feature_log_prob_
    assert not np.allclose(a, b), "sample_weight had no effect on the fit"


def test_tuning_carries_the_weights_through(dataset):
    train, val, _ = dataset
    texts, labels = train["text"].tolist()[:3000], train["label"].tolist()[:3000]
    weights = np.linspace(0.5, 2.0, len(texts))

    det = NaiveBayesDetector(tune_grid=(0.5, 1.0))
    det.fit(texts, labels, sample_weight=weights)
    det.tune(val["text"].tolist()[:500], val["label"].tolist()[:500])
    assert det._weight_cache is not None
    assert len(det.tuning_history) == 2


# interpretability


def test_top_features_are_distinctive_not_merely_frequent(fitted):
    top = fitted.top_features(k=10, class_names=DEFAULT_TASK.class_names)
    assert set(top) == {f"toward_{c}" for c in DEFAULT_TASK.class_names}
    stopwords = {"the", "and", "to", "a", "of"}
    for feats in top.values():
        tokens = {f for f, _ in feats}
        assert not tokens & stopwords, f"stopword surfaced: {tokens & stopwords}"


def test_predict_proba_is_a_distribution(fitted):
    proba = fitted.predict_proba(["muslims are scum", "nice day"])
    assert proba.shape == (2, DEFAULT_TASK.n_classes)
    assert np.allclose(proba.sum(axis=1), 1.0)
