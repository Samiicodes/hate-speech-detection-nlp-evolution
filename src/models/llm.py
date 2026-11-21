from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

from ..config import CACHE_DIR, DEFAULT_TASK, PROJECT_ROOT, TaskConfig
from ..harness import BaseDetector

PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"
CACHE_DB: Path = CACHE_DIR / "llm_responses.sqlite"

# Models that reject sampling params
NO_SAMPLING_PARAMS: tuple[str, ...] = (
    "claude-fable-5", "claude-mythos-5",
    "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
    "claude-sonnet-5",
)

# Models accepting adaptive thinking
ADAPTIVE_THINKING: tuple[str, ...] = (
    "claude-fable-5", "claude-mythos-5",
    "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-5", "claude-sonnet-4-6",
)

# Models that accept output_config={"effort": ...}. Errors on older ones.
SUPPORTS_EFFORT: tuple[str, ...] = ADAPTIVE_THINKING

# USD per million tokens
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-opus-4-8": (5.00, 25.00),
}


# prompt templates


@dataclass(frozen=True)
class PromptTemplate:

    name: str
    version: int
    system: str
    user: str
    description: str = ""
    examples: tuple[dict, ...] = ()

    @classmethod
    def load(cls, name: str, directory: Path | None = None) -> PromptTemplate:
        directory = directory or PROMPTS_DIR
        path = directory / f"{name}.yaml" if not name.endswith(".yaml") else Path(name)
        if not path.exists():
            available = ", ".join(sorted(p.stem for p in directory.glob("*.yaml")))
            raise FileNotFoundError(f"no prompt {name!r}; available: {available}")
        blob = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            name=blob["name"],
            version=int(blob["version"]),
            system=blob["system"].strip(),
            user=blob["user"],
            description=blob.get("description", "").strip(),
            examples=tuple(blob.get("examples") or ()),
        )

    def render(self, text: str) -> tuple[str, list[dict]]:
        messages: list[dict] = []
        for example in self.examples:
            messages.append(
                {"role": "user", "content": self.user.format(text=example["text"])}
            )
            messages.append({"role": "assistant", "content": example["label"]})
        messages.append({"role": "user", "content": self.user.format(text=text)})
        return self.system, messages

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "name": self.name, "version": self.version,
                "system": self.system, "user": self.user,
                "examples": list(self.examples),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# providers


@runtime_checkable
class Provider(Protocol):

    name: str
    model: str

    def complete(self, system: str, messages: list[dict]) -> tuple[str, dict]:
        ...

    def config(self) -> dict:
        ...


class AnthropicProvider:

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        effort: str = "low",
        thinking: bool = True,
        prompt_cache: bool = False,
        max_retries: int = 4,
        timeout: float = 120.0,
    ) -> None:
        import anthropic

        from ..env import require

        require("ANTHROPIC_API_KEY")   # fail with instructions, not a 401

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = thinking
        self.prompt_cache = prompt_cache
        self.supports_sampling = not model.startswith(NO_SAMPLING_PARAMS)
        self.supports_adaptive_thinking = model.startswith(ADAPTIVE_THINKING)
        self.supports_effort = model.startswith(SUPPORTS_EFFORT)
        # Run without thinking if unsupported
        self.thinking_active = thinking and self.supports_adaptive_thinking
        self._client = anthropic.Anthropic(max_retries=max_retries, timeout=timeout)

    def _base_params(self) -> dict:
        params: dict = {"model": self.model, "max_tokens": self.max_tokens}
        if self.supports_effort:
            params["output_config"] = {"effort": self.effort}
        if self.thinking_active:
            params["thinking"] = {"type": "adaptive"}
        if self.supports_sampling:
            params["temperature"] = self.temperature
        return params

    def _request_kwargs(self) -> dict:
        params = self._base_params()
        temperature = params.pop("temperature", None)
        if temperature is not None:
            params["extra_body"] = {"temperature": temperature}
        return params

    def complete(self, system: str, messages: list[dict]) -> tuple[str, dict]:
        response = self._client.messages.create(
            system=system, messages=messages, **self._request_kwargs()
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "stop_reason": response.stop_reason,
        }
        return text, usage

    # batch

    def complete_batch(
        self,
        items: list[tuple[str, str, list[dict]]],
        progress=print,
        poll_seconds: int = 30,
        chunk_size: int = 50_000,
    ) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for start in range(0, len(items), chunk_size):
            chunk = items[start : start + chunk_size]
            requests = [
                {
                    "custom_id": cid,
                    "params": {
                        **self._base_params(),
                        "system": self._cacheable_system(system),
                        "messages": messages,
                    },
                }
                for cid, system, messages in chunk
            ]
            batch = self._client.messages.batches.create(requests=requests)
            progress(
                f"batch {batch.id} submitted with {len(requests)} requests "
                f"({start + len(chunk)}/{len(items)})"
            )

            waited = 0
            while True:
                status = self._client.messages.batches.retrieve(batch.id)
                if status.processing_status == "ended":
                    break
                counts = getattr(status, "request_counts", None)
                done = (
                    (counts.succeeded + counts.errored + counts.canceled
                     + counts.expired) if counts else 0
                )
                progress(
                    f"  {status.processing_status}: {done}/{len(requests)} "
                    f"after {waited // 60}m{waited % 60:02d}s"
                )
                time.sleep(poll_seconds)
                waited += poll_seconds

            progress(f"batch {batch.id} ended after {waited // 60}m; collecting")
            for result in self._client.messages.batches.results(batch.id):
                cid = result.custom_id
                kind = result.result.type
                if kind != "succeeded":
                    out[cid] = {"error": f"{kind}: "
                                         f"{getattr(result.result, 'error', '')}"}
                    continue
                message = result.result.message
                text = "".join(
                    b.text for b in message.content if b.type == "text"
                ).strip()
                out[cid] = {
                    "text": text,
                    "stop_reason": message.stop_reason,
                    "usage": {
                        "input_tokens": message.usage.input_tokens,
                        "output_tokens": message.usage.output_tokens,
                        "stop_reason": message.stop_reason,
                    },
                }
        return out

    def _cacheable_system(self, system: str):
        if not self.prompt_cache:
            return system
        return [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]

    def count_tokens(self, system: str, messages: list[dict]) -> int:
        result = self._client.messages.count_tokens(
            model=self.model, system=system, messages=messages
        )
        return int(result.input_tokens)

    def config(self) -> dict:
        return {
            "provider": self.name,
            "model": self.model,
            "temperature_requested": self.temperature,
            "temperature_applied": self.supports_sampling,
            "temperature_note": (
                "sent to the API" if self.supports_sampling else
                f"{self.model} removes sampling parameters (400 if sent); "
                "determinism comes from the response cache instead"
            ),
            "max_tokens": self.max_tokens,
            "effort": self.effort if self.supports_effort else None,
            "effort_applied": self.supports_effort,
            "thinking_requested": self.thinking,
            "thinking_applied": self.thinking_active,
            "thinking": "adaptive" if self.thinking_active else "not sent",
            "request_shape_note": (
                f"{self.model} accepts: "
                f"temperature={self.supports_sampling}, "
                f"effort={self.supports_effort}, "
                f"adaptive_thinking={self.supports_adaptive_thinking}"
            ),
        }


class MockProvider:

    name = "mock"

    HATE_WORDS = ("scum", "vermin", "kill", "deport", "filthy", "raghead",
                  "muzzie", "nigger", "kike")
    RUDE_WORDS = ("fuck", "shit", "idiot", "stupid", "dumb", "bitch")

    def __init__(
        self,
        model: str = "mock-1",
        keywords: tuple[str, ...] | None = None,
        rude_words: tuple[str, ...] | None = None,
        n_classes: int = 3,
        malformed_every: int = 0,
    ) -> None:
        self.model = model
        self.keywords = keywords or self.HATE_WORDS
        self.rude_words = rude_words or self.RUDE_WORDS
        self.n_classes = n_classes
        self.malformed_every = malformed_every
        self._n = 0

    def complete(self, system: str, messages: list[dict]) -> tuple[str, dict]:
        self._n += 1
        usage = {"input_tokens": 0, "output_tokens": 0}
        text = messages[-1]["content"].lower()
        if self.malformed_every and self._n % self.malformed_every == 0:
            return "I'm not able to help with that.", usage

        hateful = any(k in text for k in self.keywords)
        if self.n_classes == 2:
            return ("HATE" if hateful else "NOT_HATE"), usage
        if hateful:
            return "HATESPEECH", usage
        if any(k in text for k in self.rude_words):
            return "OFFENSIVE", usage
        return "NORMAL", usage

    def config(self) -> dict:
        return {"provider": self.name, "model": self.model,
                "temperature_requested": 0.0, "temperature_applied": True,
                "n_classes": self.n_classes,
                "keywords": list(self.keywords)}


PROVIDERS = {"anthropic": AnthropicProvider, "mock": MockProvider}


def build_provider(name: str, **kwargs) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider {name!r}; have {sorted(PROVIDERS)}")
    return PROVIDERS[name](**kwargs)


# response cache


class ResponseCache:

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or CACHE_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                key        TEXT PRIMARY KEY,
                provider   TEXT NOT NULL,
                model      TEXT NOT NULL,
                prompt     TEXT NOT NULL,
                response   TEXT NOT NULL,
                usage      TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        # stop_reason added later
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(responses)")
        }
        if "stop_reason" not in columns:
            self._conn.execute("ALTER TABLE responses ADD COLUMN stop_reason TEXT")
        self._conn.commit()

    @staticmethod
    def make_key(
        provider_config: dict, prompt_fingerprint: str, text: str
    ) -> str:
        payload = json.dumps(
            {"provider": provider_config, "prompt": prompt_fingerprint, "text": text},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> tuple[str, str | None] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response, stop_reason FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return (row[0], row[1]) if row else None

    def put(
        self, key: str, provider: str, model: str, prompt: str,
        response: str, usage: dict, stop_reason: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses "
                "(key, provider, model, prompt, response, usage, created_at, "
                " stop_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (key, provider, model, prompt, response,
                 json.dumps(usage), time.time(), stop_reason),
            )
            self._conn.commit()

    def put_many(self, rows: list[tuple]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO responses "
                "(key, provider, model, prompt, response, usage, created_at, "
                " stop_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]


# label parsing

# NOT_HATE first, boundary matched
_BINARY_RE = re.compile(r"\bNOT[_\s-]?HATE\b|\bHATE\b", re.IGNORECASE)

# HATESPEECH before HATE
_THREE_CLASS_RE = re.compile(
    r"\bHATE[_\s-]?SPEECH\b|\bHATESPEECH\b|\bOFFENSIVE\b|\bNORMAL\b",
    re.IGNORECASE,
)

_THREE_CLASS_NAMES = ("hatespeech", "offensive", "normal")

# Refusal sentinel
REFUSED = "refused"

# Soft refusal patterns, narrow
_REFUSAL_PATTERNS = (
    r"\bI (?:can(?:'|no)?t|won't|will not|am not able to|'m not able to)\b"
    r"[^.]{0,60}\b(?:help|assist|answer|classify|label|complete|provide|engage)",
    r"\bI (?:cannot|can't) (?:help|assist|answer|classify|label|provide)",
    r"\bI'?m (?:sorry|afraid)[^.]{0,40}\b(?:can(?:'|no)?t|unable|not able)",
    r"\bI (?:must|have to) (?:decline|refuse)\b",
    r"\b(?:unable|not able) to (?:complete|fulfil|fulfill|process) (?:this|that)",
    r"\bI don'?t (?:feel )?comfortable\b",
    r"\bI'?m not going to\b[^.]{0,40}\b(?:classify|label|help)",
)
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(response: str, stop_reason: str | None = None) -> bool:
    if stop_reason == "refusal":
        return True
    if not response:
        return False
    # Only inspect the opening
    return bool(_REFUSAL_RE.search(response[:400]))


def parse_label(
    response: str, n_classes: int = 3, stop_reason: str | None = None
) -> tuple[int | None | str, str]:
    # Refusal checked before parsing
    if is_refusal(response, stop_reason):
        return REFUSED, "refusal"

    if not response or not response.strip():
        return None, "empty response"

    if n_classes == 2:
        match = _BINARY_RE.search(response)
        if not match:
            return None, "no HATE/NOT_HATE token found"
        return (
            (0, "not_hate") if match.group(0).upper().startswith("NOT")
            else (1, "hate")
        )

    match = _THREE_CLASS_RE.search(response)
    if not match:
        return None, "no HATESPEECH/OFFENSIVE/NORMAL token found"
    token = re.sub(r"[_\s-]", "", match.group(0)).lower()
    return _THREE_CLASS_NAMES.index(token), token


# the detector


@dataclass
class _Stats:
    n_calls: int = 0
    n_cache_hits: int = 0
    n_errors: int = 0
    n_unparseable: int = 0
    # Refusals are not unparseable
    n_refused: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    unparseable_samples: list[dict] = field(default_factory=list)
    refusal_samples: list[dict] = field(default_factory=list)
    error_samples: list[str] = field(default_factory=list)
    # index -> outcome, for the per-group refusal analysis
    refused_indices: list[int] = field(default_factory=list)
    unparseable_indices: list[int] = field(default_factory=list)


class LLMDetector(BaseDetector):

    def __init__(
        self,
        name: str | None = None,
        prompt: str = "zero_shot",
        provider: str = "anthropic",
        provider_kwargs: dict | None = None,
        task: TaskConfig = DEFAULT_TASK,
        unparseable_policy: str = "normal",
        refusal_policy: str | None = None,
        batch: bool = False,
        max_workers: int = 4,
        cache: ResponseCache | None = None,
        use_cache: bool = True,
        progress_every: int = 200,
        verbose: bool = True,
    ) -> None:
        self.task = task
        # fallback must name a class
        valid = set(task.class_names) | {"raise"}
        if task.is_binary:
            valid |= {"not_hate", "hate"}
        if unparseable_policy not in valid:
            raise ValueError(
                f"bad unparseable_policy {unparseable_policy!r}; "
                f"expected one of {sorted(valid)}"
            )
        # Refusals still need a class
        if refusal_policy is None:
            refusal_policy = (
                unparseable_policy if unparseable_policy != "raise"
                else task.class_names[-1]
            )
        if refusal_policy not in valid - {"raise"}:
            raise ValueError(
                f"bad refusal_policy {refusal_policy!r}; "
                f"expected one of {sorted(valid - {'raise'})}"
            )
        self.refusal_policy = refusal_policy
        self.batch = batch
        self.verbose = verbose

        self.template = PromptTemplate.load(prompt)
        self.provider = build_provider(provider, **(provider_kwargs or {}))
        self.name = name or f"llm_{self.template.name}_{self.provider.model}".replace(
            ".", "_"
        )
        self.unparseable_policy = unparseable_policy
        self.max_workers = max_workers
        self.use_cache = use_cache
        # is not None: empty cache is falsy
        if not use_cache:
            self.cache = None
        elif cache is not None:
            self.cache = cache
        else:
            self.cache = ResponseCache()
        self.progress_every = progress_every
        self.stats = _Stats()

    # training

    def fit(self, texts: list[str], labels: list[int]) -> None:
        self.n_train_seen = len(texts)

    # inference

    def _outcome_from_response(
        self, index: int, text: str, response: str, stop_reason: str | None
    ) -> int | None | str:
        label, reason = parse_label(response, self.task.n_classes, stop_reason)
        if label == REFUSED:
            self.stats.n_refused += 1
            self.stats.refused_indices.append(index)
            if len(self.stats.refusal_samples) < 20:
                self.stats.refusal_samples.append(
                    {"index": index, "text": text[:180],
                     "response": response[:300], "stop_reason": stop_reason}
                )
        elif label is None:
            self.stats.n_unparseable += 1
            self.stats.unparseable_indices.append(index)
            if len(self.stats.unparseable_samples) < 20:
                self.stats.unparseable_samples.append(
                    {"index": index, "text": text[:180],
                     "response": response[:300], "reason": reason}
                )
        return label

    def _cache_key(self, text: str) -> str:
        return ResponseCache.make_key(
            self.provider.config(), self.template.fingerprint(), text
        )

    def _classify_one(self, item: tuple[int, str]) -> int | None | str:
        index, text = item
        system, messages = self.template.render(text)

        key = None
        if self.cache is not None:
            key = self._cache_key(text)
            cached = self.cache.get(key)
            if cached is not None:
                self.stats.n_cache_hits += 1
                return self._outcome_from_response(index, text, *cached)

        try:
            response, usage = self.provider.complete(system, messages)
        except Exception as exc:  # noqa: BLE001 - one bad call must not kill a run
            self.stats.n_errors += 1
            if len(self.stats.error_samples) < 5:
                self.stats.error_samples.append(f"{type(exc).__name__}: {exc}")
            return None

        self.stats.n_calls += 1
        self.stats.input_tokens += usage.get("input_tokens", 0) or 0
        self.stats.output_tokens += usage.get("output_tokens", 0) or 0
        stop_reason = usage.get("stop_reason")

        if self.cache is not None and key is not None:
            self.cache.put(
                key, self.provider.name, self.provider.model,
                self.template.fingerprint(), response, usage, stop_reason,
            )
        return self._outcome_from_response(index, text, response, stop_reason)

    def predict(self, texts: list[str]) -> list[int]:
        texts = list(texts)
        if self.batch:
            raw = self._predict_batch(texts)
        elif self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                raw = list(pool.map(self._classify_one, enumerate(texts)))
        else:
            raw = [self._classify_one(item) for item in enumerate(texts)]

        if self.unparseable_policy == "raise":
            n_bad = sum(1 for r in raw if r is None or r == REFUSED)
            if n_bad:
                raise RuntimeError(
                    f"{self.name}: {n_bad} responses could not be resolved and "
                    "unparseable_policy='raise'"
                )

        fallback = self._fallback_index()
        refusal_fallback = self._refusal_fallback_index()
        return [
            refusal_fallback if r == REFUSED else fallback if r is None else r
            for r in raw
        ]

    # batch path

    def _predict_batch(self, texts: list[str]) -> list[int | None | str]:
        outcomes: list[int | None | str] = [None] * len(texts)
        pending: list[tuple[int, str]] = []

        for index, text in enumerate(texts):
            if self.cache is not None:
                cached = self.cache.get(self._cache_key(text))
                if cached is not None:
                    self.stats.n_cache_hits += 1
                    outcomes[index] = self._outcome_from_response(
                        index, text, *cached
                    )
                    continue
            pending.append((index, text))

        if not pending:
            self._log(f"all {len(texts)} responses served from cache")
            return outcomes

        self._log(
            f"{len(pending)} of {len(texts)} need calls "
            f"({self.stats.n_cache_hits} cached)"
        )
        results = self.provider.complete_batch(
            [(f"i{i}", *self.template.render(t)) for i, t in pending],
            progress=self._log,
        )

        rows = []
        for index, text in pending:
            entry = results.get(f"i{index}")
            if entry is None or entry.get("error"):
                self.stats.n_errors += 1
                if len(self.stats.error_samples) < 5:
                    self.stats.error_samples.append(
                        str((entry or {}).get("error", "no result returned"))
                    )
                continue
            response = entry["text"]
            stop_reason = entry.get("stop_reason")
            usage = entry.get("usage", {})
            self.stats.n_calls += 1
            self.stats.input_tokens += usage.get("input_tokens", 0) or 0
            self.stats.output_tokens += usage.get("output_tokens", 0) or 0
            if self.cache is not None:
                rows.append((
                    self._cache_key(text), self.provider.name,
                    self.provider.model, self.template.fingerprint(),
                    response, json.dumps(usage), time.time(), stop_reason,
                ))
            outcomes[index] = self._outcome_from_response(
                index, text, response, stop_reason
            )

        if rows and self.cache is not None:
            self.cache.put_many(rows)
        return outcomes

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"  [{self.name}] {message}", flush=True)

    def _refusal_fallback_index(self) -> int:
        policy = self.refusal_policy
        if policy in self.task.class_names:
            return self.task.index_of(policy)
        return {"not_hate": 0, "hate": 1}[policy]

    def _fallback_index(self) -> int:
        policy = self.unparseable_policy
        if policy in self.task.class_names:
            return self.task.index_of(policy)
        # binary aliases, kept so the archived binary runs still reproduce
        return {"not_hate": 0, "hate": 1}[policy]

    # harness hooks

    def params(self) -> dict:
        return {
            "prompt": {
                "name": self.template.name,
                "version": self.template.version,
                "n_examples": len(self.template.examples),
                "fingerprint": self.template.fingerprint(),
            },
            **self.provider.config(),
            "task": self.task.name,
            "n_classes": self.task.n_classes,
            "unparseable_policy": self.unparseable_policy,
            "unparseable_fallback_class": self.task.class_names[
                self._fallback_index()
            ],
            "refusal_policy": self.refusal_policy,
            "refusal_fallback_class": self.task.class_names[
                self._refusal_fallback_index()
            ],
            "batch_api": self.batch,
            "max_workers": self.max_workers,
            "cache_enabled": self.use_cache,
            "trained": False,
            "n_train_examples_seen": getattr(self, "n_train_seen", 0),
            # reported, never silently absorbed
            "n_unparseable": self.stats.n_unparseable,
            "n_refused": self.stats.n_refused,
            "n_api_errors": self.stats.n_errors,
            "n_api_calls": self.stats.n_calls,
            "n_cache_hits": self.stats.n_cache_hits,
            "unparseable_samples": self.stats.unparseable_samples,
            "refusal_samples": self.stats.refusal_samples,
            "api_error_samples": self.stats.error_samples,
            # indices for per-group refusal rate
            "refused_indices": self.stats.refused_indices,
            "unparseable_indices": self.stats.unparseable_indices,
        }

    def cost_metadata(self) -> dict:
        in_price, out_price = PRICING.get(self.provider.model, (0.0, 0.0))
        cost = (
            self.stats.input_tokens / 1e6 * in_price
            + self.stats.output_tokens / 1e6 * out_price
        )
        return {
            "model_size_bytes": None,
            "api_cost_usd": round(cost, 4),
            "input_tokens": self.stats.input_tokens,
            "output_tokens": self.stats.output_tokens,
            "size_note": "hosted API; no local parameters",
            "cost_note": (
                f"billed calls only ({self.stats.n_calls}); "
                f"{self.stats.n_cache_hits} served from cache at no cost"
            ),
        }


__all__ = [
    "PROVIDERS",
    "REFUSED",
    "AnthropicProvider",
    "is_refusal",
    "LLMDetector",
    "MockProvider",
    "PromptTemplate",
    "Provider",
    "ResponseCache",
    "build_provider",
    "parse_label",
]
