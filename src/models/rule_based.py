from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from ..config import DEFAULT_TASK, PROCESSED_DIR, TaskConfig
from ..harness import BaseDetector
from .lexicon_stoplist import FULL_STOPLIST

TOKEN_RE = re.compile(r"[a-z']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class LexiconDetector(BaseDetector):

    def __init__(
        self,
        name: str = "rule_based_lexicon",
        top_k: int = 200,
        min_freq: int = 40,
        min_len: int = 3,
        alpha: float = 0.5,
        weighted: bool = True,
        match_threshold: int = 1,
        thresholds: tuple[int, int] = (1, 2),
        tune_grid: tuple[tuple[int, int], ...] | None = None,
        task: TaskConfig = DEFAULT_TASK,
        stoplist: frozenset[str] | None = None,
    ) -> None:
        self.name = name
        self.top_k = top_k
        self.min_freq = min_freq
        self.min_len = min_len
        self.alpha = alpha
        self.weighted = weighted
        self.match_threshold = match_threshold
        self.thresholds = thresholds
        self.task = task
        self.stoplist = FULL_STOPLIST if stoplist is None else stoplist

        # candidate (t_offensive, t_hate) pairs, t_hate strictly the higher cut
        self.tune_grid = tune_grid or tuple(
            (lo, hi) for lo in (1, 2, 3) for hi in (2, 3, 4, 5, 6) if hi > lo
        )
        self.tuning_history: list[dict] = []

        self.lexicon: list[str] = []
        self.scores: dict[str, float] = {}
        self._pattern: re.Pattern | None = None

    # training

    def fit(self, texts: list[str], labels: list[int]) -> None:
        hate_index = (
            1 if self.task.is_binary else self.task.index_of("hatespeech")
        )
        hate_df: Counter = Counter()
        non_df: Counter = Counter()
        for text, label in zip(texts, labels):
            types = set(tokenize(text))
            (hate_df if label == hate_index else non_df).update(types)

        total: Counter = Counter()
        total.update(hate_df)
        total.update(non_df)

        n_hate = sum(hate_df.values())
        n_non = sum(non_df.values())
        vocab = len(total)
        a = self.alpha

        scored: list[tuple[float, str]] = []
        for word, freq in total.items():
            if freq < self.min_freq or len(word) < self.min_len:
                continue
            if word in self.stoplist:
                continue
            y_hate = hate_df[word] + a
            y_non = non_df[word] + a
            log_odds = math.log(y_hate / (n_hate + a * vocab - y_hate)) - math.log(
                y_non / (n_non + a * vocab - y_non)
            )
            if self.weighted:
                log_odds /= math.sqrt(1.0 / y_hate + 1.0 / y_non)
            scored.append((log_odds, word))

        scored.sort(reverse=True)
        top = scored[: self.top_k]
        self.lexicon = [word for _, word in top]
        self.scores = {word: score for score, word in top}
        self._compile()

    def _compile(self) -> None:
        if not self.lexicon:
            self._pattern = None
            return
        alternation = "|".join(re.escape(w) for w in sorted(self.lexicon))
        self._pattern = re.compile(rf"\b(?:{alternation})\b")

    # inference

    def hit_counts(self, texts: list[str]) -> np.ndarray:
        lex = set(self.lexicon)
        return np.array([len(set(tokenize(t)) & lex) for t in texts], dtype=int)

    def tune(self, texts: list[str], labels: list[int]) -> None:
        if self.task.is_binary or not self.lexicon:
            return
        counts = self.hit_counts(texts)
        y_true = np.asarray(labels, dtype=int)

        best, best_score = self.thresholds, -1.0
        for lo, hi in self.tune_grid:
            score = f1_score(
                y_true, self._from_counts(counts, lo, hi),
                average="macro", zero_division=0,
            )
            self.tuning_history.append(
                {"t_offensive": lo, "t_hate": hi, "val_macro_f1": float(score)}
            )
            if score > best_score:
                best, best_score = (lo, hi), score
        self.thresholds = best

    def _from_counts(self, counts: np.ndarray, lo: int, hi: int) -> np.ndarray:
        hate = self.task.index_of("hatespeech")
        offensive = self.task.index_of("offensive")
        normal = self.task.index_of("normal")
        out = np.full(len(counts), normal, dtype=int)
        out[counts >= lo] = offensive
        out[counts >= hi] = hate
        return out

    def predict(self, texts: list[str]) -> list[int]:
        if self._pattern is None:
            zero = 0 if self.task.is_binary else self.task.index_of("normal")
            return [zero] * len(texts)

        if self.task.is_binary:
            if self.match_threshold == 1:
                return [1 if self._pattern.search(t.lower()) else 0 for t in texts]
            lex = set(self.lexicon)
            return [
                int(len(set(tokenize(t)) & lex) >= self.match_threshold)
                for t in texts
            ]

        lo, hi = self.thresholds
        return [int(v) for v in self._from_counts(self.hit_counts(texts), lo, hi)]

    def explain(self, text: str) -> list[str]:
        return sorted(set(tokenize(text)) & set(self.lexicon))

    # harness hooks

    def params(self) -> dict:
        return {
            "task": self.task.name,
            "top_k": self.top_k,
            "min_freq": self.min_freq,
            "min_len": self.min_len,
            "alpha": self.alpha,
            "weighted_log_odds": self.weighted,
            "match_threshold": self.match_threshold,
            "thresholds": {
                "t_offensive": self.thresholds[0], "t_hate": self.thresholds[1],
            } if not self.task.is_binary else None,
            "tuning_history": self.tuning_history,
            "lexicon_size": len(self.lexicon),
            "stoplist_size": len(self.stoplist),
        }

    def cost_metadata(self) -> dict:
        # size on disk is the JSON word list
        blob = json.dumps({"lexicon": self.lexicon, "scores": self.scores})
        return {
            "model_size_bytes": len(blob.encode("utf-8")),
            "api_cost_usd": 0.0,
            "size_note": "serialised lexicon; no learned parameters",
        }

    def save_lexicon(self, path: Path | None = None) -> Path:
        path = path or (PROCESSED_DIR / f"{self.name}_tokens.json")
        payload = {
            "name": self.name,
            "params": self.params(),
            "lexicon": [
                {"rank": i + 1, "token": w, "score": self.scores[w]}
                for i, w in enumerate(self.lexicon)
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


__all__ = ["LexiconDetector", "tokenize"]
