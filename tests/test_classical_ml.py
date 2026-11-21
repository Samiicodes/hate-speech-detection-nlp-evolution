from __future__ import annotations

import pytest

from src.models.classical_ml import TfidfLinearDetector


@pytest.fixture(scope="module")
def small_fitted(dataset):
    train, _, _ = dataset
    det = TfidfLinearDetector(tune_grid=None)
    det.fit(train["text"].tolist()[:3000], train["label"].tolist()[:3000])
    return det


def test_predict_returns_one_class_label_per_text(small_fitted):
    from src.config import DEFAULT_TASK

    preds = small_fitted.predict(["muslims are scum", "have a nice day"])
    assert len(preds) == 2
    assert set(preds) <= set(range(DEFAULT_TASK.n_classes))


def test_multiclass_fit_is_natively_three_way(small_fitted):
    assert small_fitted.n_classes_hint == 3
    assert small_fitted.pipeline.named_steps["clf"].coef_.shape[0] == 3


def test_logistic_regression_switches_solver_for_multiclass(dataset):
    train, _, _ = dataset
    det = TfidfLinearDetector(
        classifier="logistic_regression", tune_grid=None, use_char=False
    )
    det.fit(train["text"].tolist()[:1500], train["label"].tolist()[:1500])
    assert det.pipeline.named_steps["clf"].solver == "lbfgs"
    assert len(det.predict(["muslims are scum"])) == 1


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        TfidfLinearDetector().predict(["anything"])


def test_char_ngrams_can_be_switched_off(dataset):
    train, _, _ = dataset
    det = TfidfLinearDetector(use_char=False, tune_grid=None)
    det.fit(train["text"].tolist()[:1500], train["label"].tolist()[:1500])
    names = det.pipeline.named_steps["features"].get_feature_names_out()
    assert not any(n.startswith("char__") for n in names)


def test_top_features_returns_one_list_per_class(small_fitted):
    from src.config import DEFAULT_TASK

    top = small_fitted.top_features(k=5, class_names=DEFAULT_TASK.class_names)
    assert set(top) == {f"toward_{c}" for c in DEFAULT_TASK.class_names}
    for feats in top.values():
        assert len(feats) == 5
        # each list is sorted strongest-first
        assert feats[0][1] >= feats[-1][1]


def test_tuning_only_ever_sees_train_and_val(dataset):
    train, val, _ = dataset
    det = TfidfLinearDetector(tune_grid=(0.1, 1.0))
    det.fit(train["text"].tolist()[:1500], train["label"].tolist()[:1500])
    det.tune(val["text"].tolist()[:500], val["label"].tolist()[:500])
    assert len(det.tuning_history) == 2
    assert det.C in (0.1, 1.0)


def test_is_reproducible_under_a_fixed_seed(dataset):
    train, _, test = dataset
    texts, labels = train["text"].tolist()[:1500], train["label"].tolist()[:1500]
    a, b = TfidfLinearDetector(tune_grid=None), TfidfLinearDetector(tune_grid=None)
    a.fit(texts, labels)
    b.fit(texts, labels)
    sample = test["text"].tolist()[:200]
    assert a.predict(sample) == b.predict(sample)


def test_params_record_the_feature_count(small_fitted):
    params = small_fitted.params()
    assert params["classifier"] == "linear_svc"
    assert params["n_features"] > 1000
