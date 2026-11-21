from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ..config import SEED
from ..harness import BaseDetector


class NaiveBayesDetector(BaseDetector):

    # Supports sample_weight for IPF
    supports_sample_weight = True

    def __init__(
        self,
        name: str = "naive_bayes",
        alpha: float = 1.0,
        fit_prior: bool = False,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 2,
        max_features: int | None = None,
        tune_grid: tuple[float, ...] | None = (0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
        seed: int = SEED,
    ) -> None:
        self.name = name
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.tune_grid = tune_grid
        self.seed = seed

        self.pipeline: Pipeline | None = None
        self.tuning_history: list[dict] = []
        self._train_cache: tuple[list[str], list[int]] | None = None
        self._weight_cache: np.ndarray | None = None
        self.n_classes_hint: int = 3

    # construction

    def _make_pipeline(self, alpha: float) -> Pipeline:
        return Pipeline([
            ("counts", CountVectorizer(
                analyzer="word",
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                lowercase=True,
                strip_accents="unicode",
                max_features=self.max_features,
            )),
            ("clf", MultinomialNB(alpha=alpha, fit_prior=self.fit_prior)),
        ])

    @staticmethod
    def _fit_pipeline(pipeline, texts, labels, weights) -> None:
        if weights is None:
            pipeline.fit(texts, labels)
        else:
            pipeline.fit(texts, labels, clf__sample_weight=weights)

    # training

    def fit(
        self,
        texts: list[str],
        labels: list[int],
        sample_weight: Sequence[float] | None = None,
    ) -> None:
        self._train_cache = (list(texts), list(labels))
        self._weight_cache = (
            None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        )
        self.n_classes_hint = len(set(labels))
        self.pipeline = self._make_pipeline(self.alpha)
        self._fit_pipeline(self.pipeline, texts, labels, self._weight_cache)

    def tune(self, texts: list[str], labels: list[int]) -> None:
        if not self.tune_grid or self._train_cache is None:
            return
        train_texts, train_labels = self._train_cache

        best_alpha, best_score = self.alpha, -1.0
        for alpha in self.tune_grid:
            pipe = self._make_pipeline(alpha)
            self._fit_pipeline(pipe, train_texts, train_labels, self._weight_cache)
            score = f1_score(
                labels, pipe.predict(texts), average="macro", zero_division=0
            )
            self.tuning_history.append(
                {"alpha": alpha, "val_macro_f1": float(score)}
            )
            if score > best_score:
                best_alpha, best_score = alpha, score

        self.alpha = best_alpha
        self.pipeline = self._make_pipeline(best_alpha)
        self._fit_pipeline(
            self.pipeline, train_texts, train_labels, self._weight_cache
        )

    # inference

    def predict(self, texts: list[str]) -> list[int]:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name}: fit() must be called before predict()")
        return [int(v) for v in self.pipeline.predict(texts)]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name}: fit() must be called first")
        return self.pipeline.predict_proba(texts)

    def top_features(
        self, k: int = 30, class_names: Sequence[str] | None = None
    ) -> dict[str, list[tuple[str, float]]]:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name}: fit() must be called first")
        names = np.asarray(
            self.pipeline.named_steps["counts"].get_feature_names_out()
        )
        log_prob = self.pipeline.named_steps["clf"].feature_log_prob_
        distinctive = log_prob - log_prob.mean(axis=0, keepdims=True)

        labels = list(class_names) if class_names else [
            f"class_{i}" for i in range(log_prob.shape[0])
        ]
        out: dict[str, list[tuple[str, float]]] = {}
        for i, label in enumerate(labels):
            order = np.argsort(distinctive[i])[::-1][:k]
            out[f"toward_{label}"] = [
                (str(names[j]), float(distinctive[i][j])) for j in order
            ]
        return out

    # harness hooks

    def params(self) -> dict:
        n_features = None
        if self.pipeline is not None:
            n_features = int(
                len(self.pipeline.named_steps["counts"].get_feature_names_out())
            )
        return {
            "model": "MultinomialNB",
            "features": "raw term counts (no TF-IDF)",
            "alpha": self.alpha,
            "fit_prior": self.fit_prior,
            "class_prior": "uniform" if not self.fit_prior else "empirical",
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_features": self.max_features,
            "n_features": n_features,
            "tune_grid": list(self.tune_grid) if self.tune_grid else None,
            "tuning_history": self.tuning_history,
        }

    def cost_metadata(self) -> dict:
        import pickle

        if self.pipeline is None:
            return {"api_cost_usd": 0.0}
        return {
            "model_size_bytes": len(pickle.dumps(self.pipeline)),
            "api_cost_usd": 0.0,
            "size_note": "pickled sklearn pipeline (vocabulary + log-probabilities)",
        }


__all__ = ["NaiveBayesDetector"]
