from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .config import (
    DEFAULT_SPLIT,
    DEFAULT_TASK,
    HATECHECK_TARGET,
    HATECHECK_URL,
    HATEXPLAIN_URL,
    ISLAM_TARGET,
    PROCESSED_DIR,
    RAW_DIR,
    SLICE_MIN_ANNOTATORS,
    SplitConfig,
    TaskConfig,
)

TIED = "tied"  # majority label used when all three annotators disagree


# download helpers


def _download(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def download_all(force: bool = False) -> dict[str, Path]:
    return {
        "hatexplain": _download(
            HATEXPLAIN_URL, RAW_DIR / "hatexplain_dataset.json", force
        ),
        "hatecheck": _download(
            HATECHECK_URL, RAW_DIR / "hatecheck_test_suite_cases.csv", force
        ),
    }


# HateXplain


def majority_label(labels: Iterable[str]) -> str:
    counts = Counter(labels)
    if not counts:
        return TIED
    top_label, top_count = counts.most_common(1)[0]
    return top_label if top_count >= 2 else TIED


def _normalise_target(target: str) -> str:
    return target.strip().lower()


def target_vote_counts(annotators: list[dict]) -> Counter:
    counts: Counter = Counter()
    for ann in annotators:
        for tgt in {_normalise_target(t) for t in ann.get("target", [])}:
            counts[tgt] += 1
    return counts


def load_hatexplain(
    task: TaskConfig = DEFAULT_TASK,
    path: Path | None = None,
) -> pd.DataFrame:
    path = path or (RAW_DIR / "hatexplain_dataset.json")
    if not path.exists():
        download_all()

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, dict] = json.load(fh)

    islam_key = _normalise_target(ISLAM_TARGET)
    rows: list[dict] = []
    for post_id, record in raw.items():
        annotators = record["annotators"]
        labels = [a["label"] for a in annotators]
        maj = majority_label(labels)
        votes = target_vote_counts(annotators)
        # A post can carry several targets
        groups = tuple(
            sorted(g for g, n in votes.items() if n >= SLICE_MIN_ANNOTATORS)
        )
        rows.append(
            {
                "post_id": post_id,
                "text": " ".join(record["post_tokens"]),
                "n_tokens": len(record["post_tokens"]),
                "annotator_labels": labels,
                "majority_label": maj,
                "is_tied": maj == TIED,
                "target_groups": groups,
                "islam_votes": votes.get(islam_key, 0),
                "is_islam_slice": votes.get(islam_key, 0) >= SLICE_MIN_ANNOTATORS,
                "label": task.to_label(maj),
            }
        )

    df = pd.DataFrame(rows)
    # Deterministic row order
    return df.sort_values("post_id", kind="mergesort").reset_index(drop=True)


def labelled_hatexplain(
    task: TaskConfig = DEFAULT_TASK,
    path: Path | None = None,
) -> pd.DataFrame:
    df = load_hatexplain(task=task, path=path)
    if task.drop_ties:
        df = df[~df["is_tied"]].reset_index(drop=True)
    return df


def hatexplain_summary(df: pd.DataFrame) -> dict[str, int]:
    slice_df = df[df["is_islam_slice"]]
    counts = df["majority_label"].value_counts()
    slice_counts = slice_df["majority_label"].value_counts()
    return {
        "hatexplain_total": int(len(df)),
        "majority_normal": int(counts.get("normal", 0)),
        "majority_hatespeech": int(counts.get("hatespeech", 0)),
        "majority_offensive": int(counts.get("offensive", 0)),
        "majority_tied": int(counts.get(TIED, 0)),
        "islam_slice_total": int(len(slice_df)),
        "islam_slice_hatespeech": int(slice_counts.get("hatespeech", 0)),
        "islam_slice_offensive": int(slice_counts.get("offensive", 0)),
        "islam_slice_normal": int(slice_counts.get("normal", 0)),
        "islam_slice_tied": int(slice_counts.get(TIED, 0)),
    }


# splits


@dataclass(frozen=True)
class Splits:

    train: list[str]
    val: list[str]
    test: list[str]
    config: SplitConfig

    def as_dict(self) -> dict:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "config": {
                "train_frac": self.config.train_frac,
                "val_frac": self.config.val_frac,
                "test_frac": self.config.test_frac,
                "seed": self.config.seed,
                "stratify_on_slice": self.config.stratify_on_slice,
            },
        }


def _stratum_key(row: pd.Series, stratify_on_slice: bool) -> str:
    if stratify_on_slice:
        return f"{row['label']}|{int(row['is_islam_slice'])}"
    return str(row["label"])


def make_splits(df: pd.DataFrame, cfg: SplitConfig = DEFAULT_SPLIT) -> Splits:
    strata = df.apply(lambda r: _stratum_key(r, cfg.stratify_on_slice), axis=1)
    rng = np.random.default_rng(cfg.seed)

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []

    for key in sorted(strata.unique()):
        ids = df.loc[strata == key, "post_id"].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        n = len(ids)
        n_train = int(round(n * cfg.train_frac))
        n_val = int(round(n * cfg.val_frac))
        # any rounding slack falls to test, so nothing is dropped
        train.extend(ids[:n_train].tolist())
        val.extend(ids[n_train : n_train + n_val].tolist())
        test.extend(ids[n_train + n_val :].tolist())

    return Splits(sorted(train), sorted(val), sorted(test), cfg)


def save_splits(splits: Splits, path: Path | None = None) -> Path:
    path = path or (PROCESSED_DIR / splits.config.filename)
    path.write_text(json.dumps(splits.as_dict(), indent=2), encoding="utf-8")
    return path


def load_splits(path: Path | None = None) -> Splits:
    path = path or (PROCESSED_DIR / DEFAULT_SPLIT.filename)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/prepare_data.py first"
        )
    blob = json.loads(path.read_text(encoding="utf-8"))
    cfg = SplitConfig(**blob["config"])
    return Splits(blob["train"], blob["val"], blob["test"], cfg)


def apply_splits(
    df: pd.DataFrame, splits: Splits
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indexed = df.set_index("post_id", drop=False)
    return (
        indexed.loc[splits.train].reset_index(drop=True),
        indexed.loc[splits.val].reset_index(drop=True),
        indexed.loc[splits.test].reset_index(drop=True),
    )


def get_dataset(
    task: TaskConfig = DEFAULT_TASK, cfg: SplitConfig = DEFAULT_SPLIT
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = labelled_hatexplain(task=task)
    splits = load_splits(PROCESSED_DIR / cfg.filename)
    return apply_splits(df, splits)


# HateCheck


def load_hatecheck(path: Path | None = None) -> pd.DataFrame:
    path = path or (RAW_DIR / "hatecheck_test_suite_cases.csv")
    if not path.exists():
        download_all()
    df = pd.read_csv(path)
    # label_gold is "hateful"/"non-hateful"; map onto the same binary target
    df["label"] = (df["label_gold"] == "hateful").astype(int)
    return df


def hatecheck_muslim(path: Path | None = None) -> pd.DataFrame:
    df = load_hatecheck(path)
    sub = df[df["target_ident"] == HATECHECK_TARGET].reset_index(drop=True)
    return sub


def hatecheck_summary(df: pd.DataFrame) -> dict[str, int]:
    return {
        "hatecheck_muslim_total": int(len(df)),
        "hatecheck_muslim_hateful": int((df["label"] == 1).sum()),
        "hatecheck_muslim_nonhateful": int((df["label"] == 0).sum()),
    }


# stubs for the implicit-hate corpora


def load_toxigen() -> pd.DataFrame:
    raise NotImplementedError(
        "ToxiGen is a planned addition for the under-detection analysis; "
        "set config.TOXIGEN_URL and implement this when needed."
    )


def load_latent_hatred() -> pd.DataFrame:
    raise NotImplementedError(
        "Latent Hatred is a planned addition for the under-detection "
        "analysis; set config.LATENT_HATRED_URL and implement this."
    )


__all__ = [
    "TIED",
    "Splits",
    "apply_splits",
    "download_all",
    "get_dataset",
    "hatecheck_muslim",
    "hatecheck_summary",
    "hatexplain_summary",
    "labelled_hatexplain",
    "load_hatecheck",
    "load_hatexplain",
    "load_splits",
    "majority_label",
    "make_splits",
    "save_splits",
    "target_vote_counts",
]
