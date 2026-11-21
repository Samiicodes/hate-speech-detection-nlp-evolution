from __future__ import annotations

import pytest

from src.harness import Detector
from src.models.llm import (
    LLMDetector,
    MockProvider,
    PromptTemplate,
    ResponseCache,
    build_provider,
    parse_label,
)


# label parsing


@pytest.mark.parametrize(
    "response,expected",
    [
        ("HATE", 1),
        ("NOT_HATE", 0),
        ("not_hate", 0),
        ("  hate  ", 1),
        ("NOT HATE", 0),
        ("not-hate", 0),
        ("Answer: HATE", 1),
        ("Answer: NOT_HATE", 0),
        ("HATE.", 1),
        ("The answer is NOT_HATE because it is merely rude.", 0),
        ("**HATE**", 1),
    ],
)
def test_binary_parse_reads_the_obvious_forms(response, expected):
    assert parse_label(response, n_classes=2)[0] == expected


def test_not_hate_is_never_read_as_hate():
    for form in ("NOT_HATE", "NOT HATE", "not_hate", "Answer: NOT_HATE"):
        assert parse_label(form, n_classes=2)[0] == 0, f"{form} misread as HATE"


@pytest.mark.parametrize(
    "response,expected",
    [
        ("HATESPEECH", 0),
        ("OFFENSIVE", 1),
        ("NORMAL", 2),
        ("hatespeech", 0),
        ("  offensive  ", 2 - 1),
        ("HATE_SPEECH", 0),
        ("HATE SPEECH", 0),
        ("hate-speech", 0),
        ("Answer: NORMAL", 2),
        ("**OFFENSIVE**", 1),
        ("NORMAL.", 2),
        ("The label is HATESPEECH because it attacks a group.", 0),
    ],
)
def test_three_class_parse_reads_the_obvious_forms(response, expected):
    assert parse_label(response, n_classes=3)[0] == expected


def test_hatespeech_is_never_truncated_to_hate():
    for form in ("HATESPEECH", "HATE_SPEECH", "HATE SPEECH", "hate-speech"):
        assert parse_label(form, n_classes=3)[0] == 0, f"{form} misparsed"


def test_three_class_severity_ordering_is_preserved():
    assert parse_label("HATESPEECH", 3)[0] < parse_label("OFFENSIVE", 3)[0]
    assert parse_label("OFFENSIVE", 3)[0] < parse_label("NORMAL", 3)[0]


@pytest.mark.parametrize(
    "response",
    ["", "   ", "I'm not able to help with that.", "unclear", "maybe?", "42"],
)
def test_unreadable_responses_return_none_not_zero(response):
    for n in (2, 3):
        label, reason = parse_label(response, n_classes=n)
        assert label is None
        assert reason


# prompt templates


def test_zero_shot_template_loads():
    t = PromptTemplate.load("zero_shot_3class")
    assert t.name == "zero_shot_3class"
    assert t.examples == ()
    for label in ("HATESPEECH", "OFFENSIVE", "NORMAL"):
        assert label in t.system


def test_few_shot_template_has_balanced_examples():
    t = PromptTemplate.load("few_shot_3class")
    labels = [e["label"] for e in t.examples]
    assert len(t.examples) == 9
    for label in ("HATESPEECH", "OFFENSIVE", "NORMAL"):
        assert labels.count(label) == 3, f"{label} not evenly represented"


@pytest.mark.parametrize(
    "zero,few",
    [("zero_shot", "few_shot"), ("zero_shot_3class", "few_shot_3class")],
)
def test_prompt_pairs_share_a_byte_identical_system_message(zero, few):
    assert PromptTemplate.load(zero).system == PromptTemplate.load(few).system


def test_render_puts_examples_in_as_prior_turns():
    t = PromptTemplate.load("few_shot_3class")
    system, messages = t.render("some post")
    assert len(messages) == 19          # 9 pairs + the real question
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[-1]["role"] == "user"
    assert "some post" in messages[-1]["content"]


def test_every_few_shot_label_parses_back_to_a_class():
    for name, n in (("few_shot", 2), ("few_shot_3class", 3)):
        for example in PromptTemplate.load(name).examples:
            label, _ = parse_label(example["label"], n_classes=n)
            assert label is not None, f"{name}: {example['label']!r} unparseable"


def test_unknown_prompt_names_the_available_ones():
    with pytest.raises(FileNotFoundError, match="zero_shot"):
        PromptTemplate.load("does_not_exist")


def test_fingerprint_changes_with_the_prompt():
    a = PromptTemplate.load("zero_shot_3class")
    b = PromptTemplate.load("few_shot_3class")
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() == PromptTemplate.load("zero_shot_3class").fingerprint()


def test_binary_prompts_are_still_loadable():
    assert PromptTemplate.load("zero_shot").name == "zero_shot"
    assert len(PromptTemplate.load("few_shot").examples) == 8


# cache


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path / "c.sqlite")


def test_cache_round_trips(cache):
    cache.put("k", "mock", "m", "fp", "HATE", {})
    assert cache.get("k") == ("HATE", None)
    assert len(cache) == 1


def test_cache_preserves_stop_reason(cache):
    cache.put("k", "mock", "m", "fp", "I can't help with that.", {},
              stop_reason="refusal")
    assert cache.get("k") == ("I can't help with that.", "refusal")


def test_cache_miss_returns_none(cache):
    assert cache.get("absent") is None


def test_an_empty_cache_is_still_used(cache, tmp_path):
    assert not cache                      # empty => falsy
    det = LLMDetector(provider="mock", cache=cache, max_workers=1)
    assert det.cache is cache
    det.predict(["muslims are scum"])
    assert len(cache) == 1


def test_cache_key_covers_text_prompt_and_provider_config():
    base = ({"model": "m", "temperature_requested": 0.0}, "fp1", "a post")
    key = ResponseCache.make_key(*base)
    assert key == ResponseCache.make_key(*base)                      # stable
    assert key != ResponseCache.make_key(base[0], base[1], "other")  # text
    assert key != ResponseCache.make_key(base[0], "fp2", base[2])    # prompt
    assert key != ResponseCache.make_key(
        {"model": "m", "temperature_requested": 1.0}, base[1], base[2]
    )                                                                # params


def test_second_predict_is_served_entirely_from_cache(cache):
    det = LLMDetector(provider="mock", cache=cache, max_workers=1)
    texts = ["muslims are scum", "have a nice day"]
    expected = [0, 2]                       # hatespeech, normal
    assert det.predict(texts) == expected
    calls_after_first = det.stats.n_calls
    assert det.predict(texts) == expected
    assert det.stats.n_calls == calls_after_first      # no new calls
    assert det.stats.n_cache_hits == len(texts)


def test_mock_separates_offensive_from_hatespeech(cache):
    det = LLMDetector(provider="mock", cache=cache, max_workers=1)
    assert det.predict(["muslims are scum"]) == [0]      # hatespeech
    assert det.predict(["you are a stupid idiot"]) == [1]  # offensive
    assert det.predict(["nice weather today"]) == [2]     # normal


def test_changing_the_prompt_invalidates_the_cache(cache):
    zero = LLMDetector(prompt="zero_shot_3class", provider="mock", cache=cache,
                       max_workers=1)
    zero.predict(["muslims are scum"])
    few = LLMDetector(prompt="few_shot_3class", provider="mock", cache=cache,
                      max_workers=1)
    few.predict(["muslims are scum"])
    assert few.stats.n_cache_hits == 0
    assert few.stats.n_calls == 1


# the detector


def test_detector_satisfies_the_protocol():
    assert isinstance(LLMDetector(provider="mock", use_cache=False), Detector)


def test_fit_is_a_no_op_but_records_what_it_saw():
    det = LLMDetector(provider="mock", use_cache=False)
    det.fit(["a", "b", "c"], [0, 1, 0])
    assert det.params()["trained"] is False
    assert det.params()["n_train_examples_seen"] == 3


def test_unparseable_responses_are_counted_not_hidden(cache):
    det = LLMDetector(
        provider="mock",
        provider_kwargs={"malformed_every": 2},
        cache=cache, max_workers=1,
    )
    preds = det.predict(["a", "b", "c", "d"])
    assert len(preds) == 4
    assert set(preds) <= {0, 1, 2}
    assert det.stats.n_unparseable == 2
    assert det.params()["n_unparseable"] == 2
    assert det.params()["unparseable_samples"]


def test_unparseable_policy_controls_the_fallback(cache):
    kwargs = dict(provider="mock", provider_kwargs={"malformed_every": 1},
                  max_workers=1, use_cache=False)
    assert LLMDetector(unparseable_policy="normal", **kwargs).predict(["x"]) == [2]
    assert LLMDetector(unparseable_policy="hatespeech", **kwargs).predict(["x"]) == [0]
    assert LLMDetector(unparseable_policy="offensive", **kwargs).predict(["x"]) == [1]


def test_unparseable_fallback_class_is_named_in_params():
    det = LLMDetector(provider="mock", unparseable_policy="normal",
                      use_cache=False)
    assert det.params()["unparseable_fallback_class"] == "normal"


def test_raise_policy_stops_the_run():
    det = LLMDetector(
        provider="mock", provider_kwargs={"malformed_every": 1},
        unparseable_policy="raise", max_workers=1, use_cache=False,
    )
    with pytest.raises(RuntimeError, match="could not be resolved"):
        det.predict(["x"])


def test_bad_policy_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unparseable_policy"):
        LLMDetector(provider="mock", unparseable_policy="whatever")


def test_temperature_is_logged_even_when_the_model_rejects_it():
    provider = _provider("claude-opus-5")
    cfg = provider.config()
    assert cfg["temperature_requested"] == 0.0
    assert cfg["temperature_applied"] is False
    assert "400" in cfg["temperature_note"]
    assert "temperature" not in provider._request_kwargs()


def _provider(model: str, thinking: bool = True):
    from src.models.llm import (
        ADAPTIVE_THINKING,
        NO_SAMPLING_PARAMS,
        SUPPORTS_EFFORT,
        AnthropicProvider,
    )

    p = AnthropicProvider.__new__(AnthropicProvider)
    p.model = model
    p.temperature = 0.0
    p.max_tokens = 1024
    p.effort = "low"
    p.thinking = thinking
    p.supports_sampling = not model.startswith(NO_SAMPLING_PARAMS)
    p.supports_adaptive_thinking = model.startswith(ADAPTIVE_THINKING)
    p.supports_effort = model.startswith(SUPPORTS_EFFORT)
    p.thinking_active = thinking and p.supports_adaptive_thinking
    return p


def test_older_models_are_not_sent_adaptive_thinking_or_effort():
    kwargs = _provider("claude-haiku-4-5")._request_kwargs()
    assert "thinking" not in kwargs
    assert "output_config" not in kwargs
    # SDK dropped the keyword
    assert "temperature" not in kwargs
    assert kwargs["extra_body"] == {"temperature": 0.0}


def test_temperature_is_never_passed_as_a_bare_sdk_keyword():
    import inspect

    import anthropic

    params = inspect.signature(anthropic.Anthropic.__init__).parameters
    assert "api_key" in params        # sanity: SDK shape as expected
    for model in ("claude-haiku-4-5", "claude-opus-5"):
        assert "temperature" not in _provider(model)._request_kwargs()


def test_batch_params_carry_temperature_directly():
    params = _provider("claude-haiku-4-5")._base_params()
    assert params["temperature"] == 0.0
    assert "extra_body" not in params


def test_current_models_get_thinking_and_effort():
    kwargs = _provider("claude-opus-5")._request_kwargs()
    assert kwargs["thinking"] == {"type": "adaptive"}
    assert kwargs["output_config"] == {"effort": "low"}
    assert "temperature" not in kwargs         # sampling is NOT accepted there


def test_no_thinking_flag_is_honoured_where_thinking_is_supported():
    kwargs = _provider("claude-opus-5", thinking=False)._request_kwargs()
    assert "thinking" not in kwargs
    assert "output_config" in kwargs


def test_config_records_what_was_actually_sent():
    cfg = _provider("claude-haiku-4-5").config()
    assert cfg["thinking_requested"] is True
    assert cfg["thinking_applied"] is False
    assert cfg["effort_applied"] is False
    assert cfg["temperature_applied"] is True
    assert "claude-haiku-4-5" in cfg["request_shape_note"]


def test_temperature_is_sent_to_models_that_accept_it():
    provider = _provider("claude-haiku-4-5", thinking=False)
    assert provider._request_kwargs()["extra_body"]["temperature"] == 0.0
    assert provider._base_params()["temperature"] == 0.0
    assert provider.config()["temperature_applied"] is True


def test_api_errors_do_not_kill_the_run(cache):
    class _Broken:
        name, model = "broken", "b"

        def complete(self, system, messages):
            raise RuntimeError("connection reset")

        def config(self):
            return {"provider": "broken", "model": "b"}

    det = LLMDetector(provider="mock", cache=cache, max_workers=1)
    det.provider = _Broken()
    preds = det.predict(["a", "b"])
    assert preds == [2, 2]                 # fall back to the normal class
    assert det.stats.n_errors == 2
    assert det.params()["api_error_samples"]


def test_cost_metadata_reports_zero_for_the_mock():
    det = LLMDetector(provider="mock", use_cache=False, max_workers=1)
    det.predict(["a"])
    cost = det.cost_metadata()
    assert cost["api_cost_usd"] == 0.0
    assert cost["model_size_bytes"] is None


def test_build_provider_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown provider"):
        build_provider("nope")


# refusals


@pytest.mark.parametrize(
    "response",
    [
        "I can't help with classifying that content.",
        "I cannot assist with this request.",
        "I'm sorry, but I'm not able to complete this.",
        "I must decline.",
        "I don't feel comfortable labelling this.",
        "I'm not going to classify this post.",
        "I am not able to answer that.",
    ],
)
def test_soft_refusals_are_detected(response):
    from src.models.llm import is_refusal

    assert is_refusal(response), f"{response!r} not recognised as a refusal"


def test_structured_refusal_is_detected_regardless_of_text():
    from src.models.llm import is_refusal

    assert is_refusal("anything at all", stop_reason="refusal")


@pytest.mark.parametrize(
    "response",
    [
        "HATESPEECH",
        "NORMAL",
        "OFFENSIVE",
        "This is HATESPEECH because it attacks a protected group.",
        "The post says the author cannot help Muslims, but it is NORMAL.",
    ],
)
def test_ordinary_answers_are_not_read_as_refusals(response):
    from src.models.llm import is_refusal

    assert not is_refusal(response), f"{response!r} misread as a refusal"


def test_refusal_is_its_own_outcome_not_unparseable():
    from src.models.llm import REFUSED

    label, reason = parse_label("I can't help with that.", 3)
    assert label == REFUSED
    assert reason == "refusal"


def test_refusals_and_unparseables_are_counted_separately(cache):
    class _Mixed:
        name, model = "mixed", "m"

        def __init__(self):
            self.n = 0

        def complete(self, system, messages):
            self.n += 1
            if self.n % 3 == 1:
                return "I can't help with that.", {"stop_reason": "end_turn"}
            if self.n % 3 == 2:
                return "hmm, unclear", {"stop_reason": "end_turn"}
            return "NORMAL", {"stop_reason": "end_turn"}

        def config(self):
            return {"provider": "mixed", "model": "m"}

    det = LLMDetector(provider="mock", cache=cache, max_workers=1)
    det.provider = _Mixed()
    det.predict(["a", "b", "c", "d", "e", "f"])
    assert det.stats.n_refused == 2
    assert det.stats.n_unparseable == 2
    p = det.params()
    assert p["n_refused"] == 2
    assert p["n_unparseable"] == 2
    assert p["refusal_samples"] and p["unparseable_samples"]


def test_refused_indices_are_recorded_for_the_group_breakdown(cache):
    class _RefuseFirst:
        name, model = "rf", "m"

        def __init__(self):
            self.n = 0

        def complete(self, system, messages):
            self.n += 1
            if self.n == 1:
                return "I cannot help with that.", {}
            return "NORMAL", {}

        def config(self):
            return {"provider": "rf", "model": "m"}

    det = LLMDetector(provider="mock", cache=cache, max_workers=1)
    det.provider = _RefuseFirst()
    det.predict(["a", "b", "c"])
    assert det.stats.refused_indices == [0]


def test_refusal_policy_defaults_to_the_unparseable_policy():
    det = LLMDetector(provider="mock", unparseable_policy="offensive",
                      use_cache=False)
    assert det.refusal_policy == "offensive"
    assert det.params()["refusal_fallback_class"] == "offensive"


def test_mock_provider_is_deterministic():
    a, b = MockProvider(), MockProvider()
    system, messages = PromptTemplate.load("zero_shot_3class").render(
        "they are scum"
    )
    assert a.complete(system, messages) == b.complete(system, messages)


def test_binary_detector_still_works_end_to_end(cache):
    from src.config import TASK_BINARY

    det = LLMDetector(
        prompt="zero_shot", provider="mock",
        provider_kwargs={"n_classes": 2}, task=TASK_BINARY,
        unparseable_policy="not_hate", cache=cache, max_workers=1,
    )
    assert det.predict(["muslims are scum", "nice day"]) == [1, 0]
