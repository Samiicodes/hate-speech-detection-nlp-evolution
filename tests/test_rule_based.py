from __future__ import annotations

import numpy as np
import pytest

from src.models.lexicon_stoplist import (
    FUNCTION_WORDS,
    IDENTITY_TERMS,
    INNOCUOUS_TERMS,
)
from src.models.rule_based import LexiconDetector, tokenize

# Corrected pilot figures
PILOT = {
    "recall": 0.27,
    "false_alarm": 0.243,
    "counter_quote_nh_flag_rate": 0.58,
    "threat_recall": 0.308,
}

# Violence terms, not innocuous
CORRECTED_VIOLENCE_TERMS = ("die", "murder")


HATE, OFFENSIVE, NORMAL = 0, 1, 2


@pytest.fixture(scope="module")
def fitted(dataset):
    train, val, _ = dataset
    det = LexiconDetector()
    det.fit(train["text"].tolist(), train["label"].tolist())
    det.tune(val["text"].tolist(), val["label"].tolist())
    return det


@pytest.fixture(scope="module")
def hatecheck_predictions(fitted, hatecheck):
    from src.config import DEFAULT_TASK
    from src.harness import to_hatecheck_binary

    raw = fitted.predict(hatecheck["test_case"].tolist())
    return to_hatecheck_binary(raw, DEFAULT_TASK, lenient=False)


@pytest.fixture(scope="module")
def hatecheck_predictions_lenient(fitted, hatecheck):
    from src.config import DEFAULT_TASK
    from src.harness import to_hatecheck_binary

    raw = fitted.predict(hatecheck["test_case"].tolist())
    return to_hatecheck_binary(raw, DEFAULT_TASK, lenient=True)


# tokenizer


def test_tokenizer_lowercases_and_keeps_apostrophes():
    assert tokenize("They're SCUM!") == ["they're", "scum"]


def test_tokenizer_drops_digits_and_punctuation():
    assert tokenize("h4te 2 you <user>") == ["h", "te", "you", "user"]


# lexicon construction


def test_lexicon_respects_the_top_k_cap(dataset):
    train, _, _ = dataset
    det = LexiconDetector(top_k=25)
    det.fit(train["text"].tolist(), train["label"].tolist())
    assert len(det.lexicon) == 25


def test_lexicon_contains_no_stoplisted_tokens(fitted):
    assert not set(fitted.lexicon) & FUNCTION_WORDS
    assert not set(fitted.lexicon) & INNOCUOUS_TERMS


def test_lexicon_contains_no_protected_group_identity_terms(fitted):
    assert not set(fitted.lexicon) & IDENTITY_TERMS
    for term in ("muslim", "muslims", "islam", "jews", "blacks"):
        assert term not in fitted.lexicon


def test_lexicon_surfaces_known_slurs(fitted):
    for slur in ("nigger", "kike", "muzzie", "raghead", "faggot"):
        assert slur in fitted.lexicon, f"{slur} missing from the lexicon"


def test_lexicon_is_deterministic(dataset):
    train, _, _ = dataset
    texts, labels = train["text"].tolist(), train["label"].tolist()
    a, b = LexiconDetector(), LexiconDetector()
    a.fit(texts, labels)
    b.fit(texts, labels)
    assert a.lexicon == b.lexicon


def test_lexicon_is_derived_from_training_data_only(dataset):
    train, _, _ = dataset
    full, half = LexiconDetector(), LexiconDetector()
    full.fit(train["text"].tolist(), train["label"].tolist())
    half.fit(train["text"].tolist()[:2000], train["label"].tolist()[:2000])
    assert full.lexicon != half.lexicon


# prediction behaviour


def test_one_hit_is_offensive_two_is_hatespeech(fitted):
    assert fitted.thresholds == (1, 2)
    assert fitted.predict(["you are scum"]) == [OFFENSIVE]        # 1 hit
    assert fitted.predict(["you filthy scum"]) == [HATE]          # 2 hits


def test_predict_is_clean_on_neutral_text(fitted):
    assert fitted.predict(["I live with two Muslims."]) == [NORMAL]
    assert fitted.predict(["I love Muslims."]) == [NORMAL]


def test_hit_counts_are_distinct_types_not_occurrences(fitted):
    assert fitted.hit_counts(["scum scum scum"])[0] == 1


def test_thresholds_are_tuned_on_validation_only(dataset):
    train, val, test = dataset
    det = LexiconDetector(thresholds=(3, 6))
    det.fit(train["text"].tolist(), train["label"].tolist())
    before = det.thresholds
    det.tune(val["text"].tolist(), val["label"].tolist())
    assert det.tuning_history                      # a sweep actually happened
    assert det.thresholds != before or before in det.tune_grid
    # every candidate scored is a valid ordered pair
    for row in det.tuning_history:
        assert row["t_hate"] > row["t_offensive"]


def test_higher_thresholds_predict_less_hate(dataset):
    train, _, _ = dataset
    texts = ["muslims are filthy scum", "you are scum"]
    lenient = LexiconDetector(thresholds=(1, 2))
    strict = LexiconDetector(thresholds=(2, 4))
    for det in (lenient, strict):
        det.fit(train["text"].tolist(), train["label"].tolist())
    assert sum(p == HATE for p in strict.predict(texts)) <= sum(
        p == HATE for p in lenient.predict(texts)
    )


def test_explain_names_the_token_that_fired(fitted):
    assert "scum" in fitted.explain("Muslims are scum.")


# the HateCheck regression


def test_lenient_hatecheck_recall_matches_the_binary_run(
    hatecheck_predictions_lenient, hatecheck
):
    recall = hatecheck_predictions_lenient[hatecheck["label"] == 1].mean()
    assert 0.20 <= recall <= 0.32, (
        f"lenient recall {recall:.3f} outside the band; corrected pilot "
        f"reported {PILOT['recall']:.2f}"
    )


def test_lenient_hatecheck_false_alarm_matches_the_binary_run(
    hatecheck_predictions_lenient, hatecheck
):
    fa = hatecheck_predictions_lenient[hatecheck["label"] == 0].mean()
    assert 0.20 <= fa <= 0.36, (
        f"lenient false-alarm {fa:.3f} outside the band; corrected pilot "
        f"reported {PILOT['false_alarm']:.3f}"
    )


def test_strict_recall_is_far_below_lenient(
    hatecheck_predictions, hatecheck_predictions_lenient, hatecheck
):
    hateful = hatecheck["label"] == 1
    strict = hatecheck_predictions[hateful].mean()
    lenient = hatecheck_predictions_lenient[hateful].mean()
    assert strict < lenient / 2, (
        f"strict {strict:.3f} vs lenient {lenient:.3f} -- expected a large gap"
    )


def test_violence_terms_are_in_the_lexicon_not_the_stoplist(fitted):
    for term in CORRECTED_VIOLENCE_TERMS:
        assert term in fitted.lexicon, (
            f"{term!r} missing from the lexicon -- it is a violence term and "
            "must not be stoplisted as innocuous"
        )


def test_threat_functionality_recall(hatecheck_predictions_lenient, hatecheck):
    mask = hatecheck["functionality"].isin(
        ["threat_dir_h", "threat_norm_h"]
    ).to_numpy()
    recall = hatecheck_predictions_lenient[mask].mean()
    assert 0.20 <= recall <= 0.40, (
        f"threat_* recall {recall:.3f} outside the band; corrected pilot "
        f"reported {PILOT['threat_recall']:.3f}"
    )


def test_false_alarms_concentrate_in_quoted_counter_speech(
    hatecheck_predictions_lenient, hatecheck
):
    mask = (hatecheck["functionality"] == "counter_quote_nh").to_numpy()
    rate = hatecheck_predictions_lenient[mask].mean()
    assert 0.45 <= rate <= 0.70, (
        f"counter_quote_nh flag rate {rate:.3f}; pilot reported "
        f"{PILOT['counter_quote_nh_flag_rate']:.2f}"
    )


@pytest.mark.parametrize("functionality", ["ident_pos_nh", "ident_neutral_nh"])
def test_no_false_alarms_on_plain_identity_mentions(
    hatecheck_predictions_lenient, hatecheck, functionality
):
    mask = (hatecheck["functionality"] == functionality).to_numpy()
    rate = hatecheck_predictions_lenient[mask].mean()
    assert rate <= 0.05, f"{functionality} flag rate {rate:.3f}, expected ~0"


def test_obfuscated_hate_is_missed(fitted):
    assert fitted.predict(["I h4te Muslims"]) == [NORMAL]
    assert fitted.predict(["I h a t e Muslims"]) == [NORMAL]


def test_implicit_hate_is_missed(fitted):
    assert fitted.predict(["Muslims belong in a zoo."]) == [NORMAL]


def test_severity_separation_is_poor_on_the_test_split(fitted, dataset):
    _, _, test = dataset
    preds = np.asarray(fitted.predict(test["text"].tolist()))
    truth = test["label"].to_numpy()

    hate_to_off = (preds[truth == HATE] == OFFENSIVE).mean()
    off_to_hate = (preds[truth == OFFENSIVE] == HATE).mean()
    assert hate_to_off > 0.20, "expected substantial hate->offensive confusion"
    assert off_to_hate > 0.20, "expected substantial offensive->hate confusion"


# harness hooks


def test_params_and_cost_are_reported(fitted):
    params = fitted.params()
    assert params["min_freq"] == 40
    assert params["top_k"] == 200
    assert params["lexicon_size"] == len(fitted.lexicon)
    assert fitted.cost_metadata()["model_size_bytes"] > 0
