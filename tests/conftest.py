from __future__ import annotations

import pytest

from src import config, data


def _require_data() -> None:
    raw = config.RAW_DIR / "hatexplain_dataset.json"
    if not raw.exists():
        pytest.skip("corpora not downloaded -- run scripts/prepare_data.py first")


@pytest.fixture(scope="session")
def hatexplain_all():
    _require_data()
    return data.load_hatexplain()


@pytest.fixture(scope="session")
def hatexplain_labelled():
    _require_data()
    return data.labelled_hatexplain()


@pytest.fixture(scope="session")
def hatecheck():
    _require_data()
    return data.hatecheck_muslim()


@pytest.fixture(scope="session")
def splits():
    if not (config.PROCESSED_DIR / config.DEFAULT_SPLIT.filename).exists():
        pytest.skip("splits not built -- run scripts/prepare_data.py first")
    return data.load_splits()


@pytest.fixture(scope="session")
def dataset(splits, hatexplain_labelled):
    return data.apply_splits(hatexplain_labelled, splits)
