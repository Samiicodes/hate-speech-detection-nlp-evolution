from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .config import DEFAULT_TASK, PROCESSED_DIR, RESULTS_DIR, SEED, TaskConfig

CV_DIR: Path = RESULTS_DIR / "crossval"
CV_DIR.mkdir(parents=True, exist_ok=True)

FOLDS_FILE: Path = PROCESSED_DIR / "cv_folds.json"

# Fraction of each fold's training portion held back for tuning.
INNER_VAL_FRAC: float = 0.15


# folds


@dataclass(frozen=True)
class Folds:

    post_ids: list[str]
    fold: list[int]
    n_splits: int
    seed: int
    stratify: str

    def as_dict(self) -> dict:
        return {
            "post_ids": self.post_ids,
            "fold": self.fold,
            "n_splits": self.n_splits,
            "seed": self.seed,
            "stratify": self.stratify,
        }

    def assignment(self) -> dict[str, int]:
        return dict(zip(self.post_ids, self.fold))


def make_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = SEED,
    stratify_on_slice: bool = True,
) -> Folds:
    from sklearn.model_selection import StratifiedKFold

    df = df.sort_values("post_id", kind="mergesort").reset_index(drop=True)
    if stratify_on_slice:
        strata = (
            df["label"].astype(str) + "|" + df["is_islam_slice"].astype(int).astype(str)
        )
    else:
        strata = df["label"].astype(str)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for i, (_, test_idx) in enumerate(splitter.split(df, strata)):
        fold[test_idx] = i

    return Folds(
        post_ids=df["post_id"].tolist(),
        fold=fold.tolist(),
        n_splits=n_splits,
        seed=seed,
        stratify="label|is_islam_slice" if stratify_on_slice else "label",
    )


def save_folds(folds: Folds, path: Path | None = None) -> Path:
    path = path or FOLDS_FILE
    path.write_text(json.dumps(folds.as_dict(), indent=2), encoding="utf-8")
    return path


def load_folds(path: Path | None = None) -> Folds:
    path = path or FOLDS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/run_crossval.py --build-folds"
        )
    blob = json.loads(path.read_text(encoding="utf-8"))
    return Folds(**blob)


# out-of-fold prediction


def inner_split(
    train_df: pd.DataFrame, seed: int, val_frac: float = INNER_VAL_FRAC
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    inner_train, inner_val = train_test_split(
        train_df,
        test_size=val_frac,
        random_state=seed,
        stratify=train_df["label"],
    )
    return inner_train, inner_val


def out_of_fold_predictions(
    df: pd.DataFrame,
    folds: Folds,
    build_detector: Callable[[int, int], object],
    seed: int = SEED,
    progress: Callable[[str], None] = print,
    sample_weight: Sequence[float] | None = None,
) -> np.ndarray:
    assignment = folds.assignment()
    df = df.copy().reset_index(drop=True)
    df["_fold"] = df["post_id"].map(assignment)
    if df["_fold"].isna().any():
        missing = int(df["_fold"].isna().sum())
        raise ValueError(f"{missing} posts have no fold assignment")
    df["_fold"] = df["_fold"].astype(int)

    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        if len(weights) != len(df):
            raise ValueError(
                f"sample_weight has {len(weights)} entries for {len(df)} posts"
            )
        df["_weight"] = weights

    preds = np.full(len(df), -1, dtype=int)
    for k in range(folds.n_splits):
        held_out = df["_fold"] == k
        train_part = df[~held_out]
        inner_train, inner_val = inner_split(train_part, seed)

        detector = build_detector(k, seed)
        progress(
            f"  fold {k + 1}/{folds.n_splits}: train {len(inner_train)}, "
            f"tune {len(inner_val)}, predict {int(held_out.sum())}"
        )
        if sample_weight is None:
            detector.fit(inner_train["text"].tolist(),
                         inner_train["label"].tolist())
        else:
            if not getattr(detector, "supports_sample_weight", False):
                raise TypeError(
                    f"{type(detector).__name__} does not support sample_weight; "
                    "the correction would silently do nothing"
                )
            detector.fit(
                inner_train["text"].tolist(),
                inner_train["label"].tolist(),
                sample_weight=inner_train["_weight"].to_numpy(),
            )
        if hasattr(detector, "tune"):
            detector.tune(inner_val["text"].tolist(), inner_val["label"].tolist())

        fold_pred = detector.predict(df.loc[held_out, "text"].tolist())
        preds[held_out.to_numpy()] = np.asarray(fold_pred, dtype=int)

    if (preds < 0).any():
        raise RuntimeError("some posts never received an out-of-fold prediction")
    return preds


# storage


def oof_path(model_name: str) -> Path:
    return CV_DIR / f"{model_name}_oof.json"


def save_oof(
    model_name: str,
    df: pd.DataFrame,
    predictions: dict[int, Sequence[int]],
    folds: Folds,
    task: TaskConfig = DEFAULT_TASK,
    notes: str = "",
    extra: dict | None = None,
) -> Path:
    payload = {
        "model_name": model_name,
        "kind": "out_of_fold",
        "task": {"name": task.name, "class_names": list(task.class_names)},
        "n_splits": folds.n_splits,
        "fold_seed": folds.seed,
        "stratify": folds.stratify,
        "n_posts": int(len(df)),
        "notes": notes,
        "post_ids": df["post_id"].tolist(),
        "y_true": [int(v) for v in df["label"]],
        "seeds": sorted(predictions),
        "predictions": {
            str(seed): [int(v) for v in preds] for seed, preds in predictions.items()
        },
    }
    if extra:
        payload.update(extra)
    path = oof_path(model_name)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_oof(model_name: str) -> dict:
    path = oof_path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/run_crossval.py")
    return json.loads(path.read_text(encoding="utf-8"))


def available_oof() -> list[str]:
    return sorted(p.stem.removesuffix("_oof") for p in CV_DIR.glob("*_oof.json"))


__all__ = [
    "CV_DIR",
    "Folds",
    "available_oof",
    "inner_split",
    "load_folds",
    "load_oof",
    "make_folds",
    "oof_path",
    "out_of_fold_predictions",
    "save_folds",
    "save_oof",
]
