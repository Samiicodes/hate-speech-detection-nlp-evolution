from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from ..config import SEED
from ..harness import BaseDetector

ClassifierName = Literal["linear_svc", "logistic_regression"]


def _build_features(
    word_ngrams: tuple[int, int],
    char_ngrams: tuple[int, int],
    min_df: int,
    max_features: int | None,
    use_char: bool,
) -> FeatureUnion | TfidfVectorizer:
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngrams,
        min_df=min_df,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
        max_features=max_features,
    )
    if not use_char:
        return word
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngrams,
        min_df=min_df,
        sublinear_tf=True,
        lowercase=True,
        max_features=max_features,
    )
    return FeatureUnion([("word", word), ("char", char)])


class TfidfLinearDetector(BaseDetector):

    def __init__(
        self,
        name: str = "tfidf_linear_svc",
        classifier: ClassifierName = "linear_svc",
        C: float = 1.0,
        class_weight: str | None = "balanced",
        word_ngrams: tuple[int, int] = (1, 2),
        char_ngrams: tuple[int, int] = (3, 5),
        use_char: bool = True,
        min_df: int = 2,
        max_features: int | None = None,
        tune_grid: tuple[float, ...] | None = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0),
        seed: int = SEED,
    ) -> None:
        self.name = name
        self.classifier = classifier
        self.C = C
        self.class_weight = class_weight
        self.word_ngrams = word_ngrams
        self.char_ngrams = char_ngrams
        self.use_char = use_char
        self.min_df = min_df
        self.max_features = max_features
        self.tune_grid = tune_grid
        self.seed = seed

        self.pipeline: Pipeline | None = None
        self._train_cache: tuple[list[str], list[int]] | None = None
        self.tuning_history: list[dict] = []
        self._weight_cache = None
        # set in fit, picks the solver
        self.n_classes_hint: int = 2

    # construction

    def _make_pipeline(self, C: float) -> Pipeline:
        features = _build_features(
            self.word_ngrams, self.char_ngrams, self.min_df,
            self.max_features, self.use_char,
        )
        if self.classifier == "linear_svc":
            clf = LinearSVC(
                C=C, class_weight=self.class_weight, random_state=self.seed,
                max_iter=5000,
            )
        elif self.classifier == "logistic_regression":
            # liblinear is binary only
            solver = "liblinear" if self.n_classes_hint == 2 else "lbfgs"
            clf = LogisticRegression(
                C=C, class_weight=self.class_weight, random_state=self.seed,
                max_iter=2000, solver=solver,
            )
        else:
            raise ValueError(f"unknown classifier {self.classifier!r}")
        return Pipeline([("features", features), ("clf", clf)])

    # training

    # The harness may pass per-instance weights; see src/reweight.py.
    supports_sample_weight = True

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
        self.pipeline = self._make_pipeline(self.C)
        self._fit_pipeline(self.pipeline, texts, labels, self._weight_cache)

    @staticmethod
    def _fit_pipeline(pipeline, texts, labels, weights) -> None:
        if weights is None:
            pipeline.fit(texts, labels)
        else:
            pipeline.fit(texts, labels, clf__sample_weight=weights)

    def tune(self, texts: list[str], labels: list[int]) -> None:
        if not self.tune_grid or self._train_cache is None:
            return
        train_texts, train_labels = self._train_cache
        weights = getattr(self, "_weight_cache", None)

        best_c, best_score = self.C, -1.0
        for c in self.tune_grid:
            pipe = self._make_pipeline(c)
            self._fit_pipeline(pipe, train_texts, train_labels, weights)
            score = f1_score(labels, pipe.predict(texts), average="macro")
            self.tuning_history.append({"C": c, "val_macro_f1": float(score)})
            if score > best_score:
                best_c, best_score = c, score

        self.C = best_c
        self.pipeline = self._make_pipeline(best_c)
        self._fit_pipeline(self.pipeline, train_texts, train_labels, weights)

    # inference

    def predict(self, texts: list[str]) -> list[int]:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name}: fit() must be called before predict()")
        return [int(v) for v in self.pipeline.predict(texts)]

    def decision_scores(self, texts: list[str]) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name}: fit() must be called first")
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            proba = self.pipeline.predict_proba(texts)
            return proba[:, 1] if proba.shape[1] == 2 else proba
        return self.pipeline.decision_function(texts)

    def top_features(
        self, k: int = 30, class_names: Sequence[str] | None = None
    ) -> dict[str, list[tuple[str, float]]]:
        if self.pipeline is None:
            raise RuntimeError(f"{self.name}: fit() must be called first")
        names = np.asarray(
            self.pipeline.named_steps["features"].get_feature_names_out()
        )
        coef = np.atleast_2d(self.pipeline.named_steps["clf"].coef_)

        def _top(weights: np.ndarray, reverse: bool) -> list[tuple[str, float]]:
            order = np.argsort(weights)
            picks = order[::-1][:k] if reverse else order[:k]
            return [(str(names[i]), float(weights[i])) for i in picks]

        if coef.shape[0] == 1:
            return {
                "toward_positive": _top(coef[0], True),
                "toward_negative": _top(coef[0], False),
            }

        labels = list(class_names) if class_names else [
            f"class_{i}" for i in range(coef.shape[0])
        ]
        return {f"toward_{labels[i]}": _top(coef[i], True) for i in range(len(labels))}

    # harness hooks

    def params(self) -> dict:
        n_features = None
        if self.pipeline is not None:
            n_features = int(
                len(self.pipeline.named_steps["features"].get_feature_names_out())
            )
        return {
            "classifier": self.classifier,
            "C": self.C,
            "class_weight": self.class_weight,
            "word_ngrams": list(self.word_ngrams),
            "char_ngrams": list(self.char_ngrams) if self.use_char else None,
            "use_char": self.use_char,
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
            "size_note": "pickled sklearn pipeline (vectorisers + weights)",
        }


__all__ = ["TfidfLinearDetector"]
