from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# paths

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"           # downloaded files, untouched
PROCESSED_DIR: Path = DATA_DIR / "processed"  # split indices, derived lexicons
CACHE_DIR: Path = DATA_DIR / "cache"       # API responses, model downloads
RESULTS_DIR: Path = PROJECT_ROOT / "results"
# Archived binary results
RESULTS_BINARY_DIR: Path = PROJECT_ROOT / "results_binary"
# Per-seed runs live here
RUNS_DIR: Path = RESULTS_DIR / "runs"

for _d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, RESULTS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# reproducibility

# One seed for the project
SEED: int = 42

# data sources

HATEXPLAIN_URL: str = (
    "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
)
HATECHECK_URL: str = (
    "https://raw.githubusercontent.com/paul-rottger/hatecheck-data/main/"
    "test_suite_cases.csv"
)

# Not wired up yet
TOXIGEN_URL: str | None = None       # TODO: ToxiGen (Hartvigsen et al. 2022)
LATENT_HATRED_URL: str | None = None  # TODO: Latent Hatred (ElSherief et al. 2021)

# task definition

# HateXplain gives three annotators one of these three labels each.
HATEXPLAIN_LABELS: tuple[str, ...] = ("hatespeech", "offensive", "normal")

# The target-community string HateXplain uses for the Muslim community.
ISLAM_TARGET: str = "Islam"

# Target communities, lowercase
TARGET_GROUPS: tuple[str, ...] = (
    "african", "arab", "asian", "caucasian", "hispanic", "indian",
    "indigenous", "islam", "jewish", "christian", "hindu", "buddhism",
    "nonreligious", "homosexual", "heterosexual", "bisexual",
    "men", "women", "refugee", "immigrant", "disability", "economic",
    "minority", "asexual", "other-lifestyle",
)

# Ordered by the human gradient
ANALYSIS_GROUPS: tuple[str, ...] = (
    "jewish", "african", "hispanic", "arab", "islam",
    "asian", "homosexual", "refugee", "women", "caucasian",
)

# Reliability floor
MIN_GROUP_N: int = 200

# HateCheck names the same community differently.
HATECHECK_TARGET: str = "Muslims"

# Slice membership threshold
SLICE_MIN_ANNOTATORS: int = 2


@dataclass(frozen=True)
class TaskConfig:

    name: str
    class_names: tuple[str, ...]
    label_map: dict[str, int]
    drop_ties: bool = True
    description: str = ""

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def is_binary(self) -> bool:
        return self.n_classes == 2

    def to_label(self, majority_label: str) -> int:
        return self.label_map.get(majority_label, -1)

    def index_of(self, class_name: str) -> int:
        return self.class_names.index(class_name)


# The original design, archived under results_binary/.
TASK_BINARY = TaskConfig(
    name="hate_only",
    class_names=("not_hate", "hate"),
    label_map={"hatespeech": 1, "offensive": 0, "normal": 0},
    description="hatespeech = 1; offensive and normal = 0",
)

# Severity order, most severe first
TASK_THREE_CLASS = TaskConfig(
    name="three_class",
    class_names=("hatespeech", "offensive", "normal"),
    label_map={"hatespeech": 0, "offensive": 1, "normal": 2},
    description="hatespeech = 0; offensive = 1; normal = 2",
)

# Kept for the sensitivity check discussed in the write-up.
TASK_HATE_OR_OFFENSIVE = TaskConfig(
    name="hate_or_offensive",
    class_names=("normal", "hostile"),
    label_map={"hatespeech": 1, "offensive": 1, "normal": 0},
    description="hatespeech and offensive = 1; normal = 0",
)

DEFAULT_TASK: TaskConfig = TASK_THREE_CLASS

TASKS: dict[str, TaskConfig] = {
    t.name: t for t in (TASK_THREE_CLASS, TASK_BINARY, TASK_HATE_OR_OFFENSIVE)
}

# Hostile labels
HOSTILE_LABELS: tuple[str, ...] = ("hatespeech", "offensive")

# splits


@dataclass(frozen=True)
class SplitConfig:

    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = SEED
    stratify_on_slice: bool = True
    filename: str = "splits.json"

    def __post_init__(self) -> None:
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1.0, got {total}")


DEFAULT_SPLIT = SplitConfig()

# Class names for the active task
CLASS_NAMES: tuple[str, ...] = DEFAULT_TASK.class_names

# expected counts (pilot figures, used

EXPECTED: dict[str, int] = {
    "hatexplain_total": 20148,
    "majority_normal": 7814,
    "majority_hatespeech": 5935,
    "majority_offensive": 5480,
    "majority_tied": 919,
    "islam_slice_total": 2256,
    "islam_slice_hatespeech": 1197,
    "islam_slice_offensive": 618,
    "islam_slice_normal": 296,
    "islam_slice_tied": 145,
    "hatecheck_muslim_total": 484,
    "hatecheck_muslim_hateful": 373,
    "hatecheck_muslim_nonhateful": 111,
}


def results_path(model_name: str, runs: bool = False) -> Path:
    return (RUNS_DIR if runs else RESULTS_DIR) / f"{model_name}.json"
