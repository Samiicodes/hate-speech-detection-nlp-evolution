from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from . import config
from .config import CLASS_NAMES, DEFAULT_SPLIT, DEFAULT_TASK, SEED, TaskConfig

# The four divergence functionalities
KEY_FUNCTIONALITIES: tuple[str, ...] = (
    "ident_pos_nh",
    "ident_neutral_nh",
    "counter_quote_nh",
    "counter_ref_nh",
)

# the interface every system must implement


@runtime_checkable
class Detector(Protocol):

    name: str

    def fit(self, texts: list[str], labels: list[int]) -> None:
        ...

    def predict(self, texts: list[str]) -> list[int]:
        ...


class BaseDetector:

    name: str = "base"

    def fit(self, texts: list[str], labels: list[int]) -> None:  # noqa: D102
        return None

    def predict(self, texts: list[str]) -> list[int]:  # noqa: D102
        raise NotImplementedError

    def params(self) -> dict:
        return {}

    def cost_metadata(self) -> dict:
        return {}


# metric blocks


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str] | None = None,
) -> dict:
    names = tuple(class_names) if class_names is not None else tuple(CLASS_NAMES)
    labels = list(range(len(names)))

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Empty slice is legitimate
    if len(y_true) == 0:
        empty = [[0] * len(names) for _ in names]
        return {
            "n": 0,
            "accuracy": None,
            "macro_f1": None,
            "per_class": {
                name: {"precision": None, "recall": None, "f1": None, "support": 0}
                for name in names
            },
            "confusion_matrix": {"labels": list(names), "matrix": empty},
        }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            names[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in labels
        },
        "confusion_matrix": {"labels": list(names), "matrix": cm.tolist()},
    }

    if len(names) == 2:
        tn, fp, fn, tp = cm.ravel()
        out["confusion_matrix"].update(
            {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        )
        out["over_flag_rate"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
        out["under_detect_rate"] = float(fn / (fn + tp)) if (fn + tp) else 0.0

    return out


def _metric_fn(metric: str):
    from sklearn.metrics import precision_score, recall_score

    if metric == "accuracy":
        return accuracy_score
    if metric == "macro_f1":
        return lambda t, p: f1_score(t, p, average="macro", zero_division=0)

    # Works for any class count
    scorers = {"recall": recall_score, "precision": precision_score, "f1": f1_score}
    for name, scorer in scorers.items():
        if metric.startswith(f"{name}_"):
            cls = int(metric.split("_")[-1])
            return lambda t, p, s=scorer, c=cls: s(
                t, p, labels=[c], average="macro", zero_division=0
            )
    raise ValueError(f"unknown metric {metric!r}")


def bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    metric: str = "macro_f1",
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)
    if n == 0:
        return {"metric": metric, "point": None, "lo": None, "hi": None}

    fn = _metric_fn(metric)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    scores = np.array([fn(y_true[i], y_pred[i]) for i in idx])

    return {
        "metric": metric,
        "point": float(fn(y_true, y_pred)),
        "lo": float(np.percentile(scores, 100 * alpha / 2)),
        "hi": float(np.percentile(scores, 100 * (1 - alpha / 2))),
        "n_resamples": n_resamples,
        "alpha": alpha,
    }


def mcnemar_test(
    y_true: Sequence[int],
    pred_a: Sequence[int],
    pred_b: Sequence[int],
    exact_threshold: int = 25,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    a = np.asarray(pred_a, dtype=int) == y_true
    b = np.asarray(pred_b, dtype=int) == y_true

    n01 = int(np.sum(a & ~b))  # a right, b wrong
    n10 = int(np.sum(~a & b))  # a wrong, b right
    n_discordant = n01 + n10

    if n_discordant == 0:
        return {
            "n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0,
            "method": "degenerate (no discordant pairs)",
        }

    if n_discordant < exact_threshold:
        p = float(stats.binomtest(n01, n_discordant, 0.5).pvalue)
        statistic, method = float(min(n01, n10)), "exact binomial"
    else:
        statistic = (abs(n01 - n10) - 1) ** 2 / n_discordant
        p = float(stats.chi2.sf(statistic, df=1))
        statistic, method = float(statistic), "chi-square with continuity correction"

    return {
        "n01_a_correct_b_wrong": n01,
        "n10_a_wrong_b_correct": n10,
        "statistic": statistic,
        "p_value": p,
        "method": method,
        "accuracy_a": float(a.mean()),
        "accuracy_b": float(b.mean()),
    }


def paired_bootstrap_diff(
    y_true: Sequence[int],
    pred_a: Sequence[int],
    pred_b: Sequence[int],
    metric: str = "macro_f1",
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    n = len(y_true)
    if n == 0:
        return {"metric": metric, "point": None, "lo": None, "hi": None}

    fn = _metric_fn(metric)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = np.array([fn(y_true[i], a[i]) - fn(y_true[i], b[i]) for i in idx])

    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return {
        "metric": metric,
        "point": float(fn(y_true, a) - fn(y_true, b)),
        "lo": lo,
        "hi": hi,
        "width": hi - lo,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_resamples": n_resamples,
        "alpha": alpha,
    }


def over_flag_by_original_label(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    majority_labels: Sequence[str],
    is_slice: Sequence[bool],
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = np.asarray(majority_labels)
    slice_mask = np.asarray(is_slice, dtype=bool)

    out: dict = {}
    for bucket in ("normal", "offensive"):
        bucket_mask = (labels == bucket) & (y_true == 0)
        entry = {}
        for group, mask in (
            ("islam", bucket_mask & slice_mask),
            ("rest", bucket_mask & ~slice_mask),
        ):
            n = int(mask.sum())
            n_flagged = int(y_pred[mask].sum()) if n else 0
            entry[group] = {
                "n": n,
                "n_flagged": n_flagged,
                "over_flag_rate": (n_flagged / n) if n else None,
            }
        islam_rate = entry["islam"]["over_flag_rate"]
        rest_rate = entry["rest"]["over_flag_rate"]
        entry["gap"] = (
            None if islam_rate is None or rest_rate is None
            else islam_rate - rest_rate
        )
        out[bucket] = entry
    return out


def to_hatecheck_binary(
    y_pred: Sequence[int], task: TaskConfig, lenient: bool = False
) -> np.ndarray:
    y_pred = np.asarray(y_pred, dtype=int)
    if task.is_binary:
        return y_pred
    hostile = {task.index_of("hatespeech")}
    if lenient:
        hostile.add(task.index_of("offensive"))
    return np.isin(y_pred, list(hostile)).astype(int)


def hatecheck_report(
    df: pd.DataFrame, y_pred: Sequence[int], task: TaskConfig | None = None
) -> dict:
    task = task or DEFAULT_TASK
    raw_pred = np.asarray(y_pred, dtype=int)
    y_pred = to_hatecheck_binary(raw_pred, task, lenient=False)
    lenient_pred = to_hatecheck_binary(raw_pred, task, lenient=True)
    y_true = df["label"].to_numpy(dtype=int)
    correct = y_pred == y_true

    hateful = y_true == 1
    non_hateful = ~hateful

    per_func: dict[str, dict] = {}
    for func in sorted(df["functionality"].unique()):
        mask = (df["functionality"] == func).to_numpy()
        gold = int(y_true[mask][0])  # gold label is constant within a functionality
        n = int(mask.sum())
        n_correct = int(correct[mask].sum())
        per_func[func] = {
            "n": n,
            "gold": "hateful" if gold else "non-hateful",
            "n_correct": n_correct,
            "pass_rate": n_correct / n,
            "flag_rate": float(y_pred[mask].mean()),
            # Lenient rate too
            "flag_rate_lenient": float(lenient_pred[mask].mean()),
        }

    report = {
        "n_cases": int(len(df)),
        "n_hateful": int(hateful.sum()),
        "n_non_hateful": int(non_hateful.sum()),
        # the two headline numbers
        "recall_on_hateful": float(y_pred[hateful].mean()) if hateful.any() else 0.0,
        "false_alarm_on_non_hateful": (
            float(y_pred[non_hateful].mean()) if non_hateful.any() else 0.0
        ),
        "accuracy": float(correct.mean()),
        "mapping": "strict: only hatespeech counts as flagged",
        "per_functionality": per_func,
    }

    if not task.is_binary:
        # Downgraded, not missed
        report["lenient"] = {
            "recall_on_hateful": (
                float(lenient_pred[hateful].mean()) if hateful.any() else 0.0
            ),
            "false_alarm_on_non_hateful": (
                float(lenient_pred[non_hateful].mean()) if non_hateful.any() else 0.0
            ),
            "mapping": "lenient: hatespeech or offensive counts as flagged",
        }
        report["downgraded_share_of_hateful"] = (
            report["lenient"]["recall_on_hateful"] - report["recall_on_hateful"]
        )

    return report


def key_metrics(
    islam: dict, non_islam: dict, hatecheck: dict, task: TaskConfig | None = None
) -> dict:
    task = task or DEFAULT_TASK
    per_func = hatecheck.get("per_functionality", {})

    def _gap(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else a - b

    out = {
        "macro_f1_islam": islam["macro_f1"],
        "macro_f1_rest": non_islam["macro_f1"],
        "macro_f1_gap": _gap(islam["macro_f1"], non_islam["macro_f1"]),
        "hatecheck_key_functionalities": {
            func: (per_func[func]["pass_rate"] if func in per_func else None)
            for func in KEY_FUNCTIONALITIES
        },
    }
    # Binary notion only
    if task.is_binary:
        out.update({
            "over_flag_islam": islam.get("over_flag_rate"),
            "over_flag_rest": non_islam.get("over_flag_rate"),
            "over_flag_gap": _gap(
                islam.get("over_flag_rate"), non_islam.get("over_flag_rate")
            ),
        })
    return out


# the evaluation entry point


@dataclass
class EvalInputs:

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    hatecheck: pd.DataFrame
    task: TaskConfig = field(default=DEFAULT_TASK)


def evaluate(
    detector: Detector,
    inputs: EvalInputs,
    *,
    seed: int = SEED,
    n_bootstrap: int = 2000,
    write: bool = True,
    notes: str = "",
    to_runs_dir: bool = False,
) -> dict:
    name = detector.name

    # fit, timed
    t0 = time.perf_counter()
    detector.fit(inputs.train["text"].tolist(), inputs.train["label"].tolist())
    train_seconds = time.perf_counter() - t0

    # optional validation-set tuning, still never
    val_seconds = 0.0
    if hasattr(detector, "tune"):
        t0 = time.perf_counter()
        detector.tune(inputs.val["text"].tolist(), inputs.val["label"].tolist())
        val_seconds = time.perf_counter() - t0

    # predict on the HateXplain test
    test_texts = inputs.test["text"].tolist()
    t0 = time.perf_counter()
    test_pred = list(detector.predict(test_texts))
    infer_seconds = time.perf_counter() - t0

    if len(test_pred) != len(test_texts):
        raise ValueError(
            f"{name}.predict returned {len(test_pred)} labels for "
            f"{len(test_texts)} texts"
        )

    test_true = inputs.test["label"].tolist()
    is_slice = inputs.test["is_islam_slice"].to_numpy(dtype=bool)
    pred_arr = np.asarray(test_pred, dtype=int)
    true_arr = np.asarray(test_true, dtype=int)

    # predict on HateCheck
    hc_texts = inputs.hatecheck["test_case"].tolist()
    t0 = time.perf_counter()
    hc_pred = list(detector.predict(hc_texts))
    hc_seconds = time.perf_counter() - t0

    # validation-split metrics, useful for spotting
    val_pred = list(detector.predict(inputs.val["text"].tolist()))

    names = inputs.task.class_names
    slice_islam = classification_metrics(
        true_arr[is_slice], pred_arr[is_slice], names
    )
    slice_rest = classification_metrics(
        true_arr[~is_slice], pred_arr[~is_slice], names
    )
    hc_report = hatecheck_report(inputs.hatecheck, hc_pred, inputs.task)

    # per-target-group severity measures; only defined for the three-class task
    group_block: dict = {}
    if not inputs.task.is_binary:
        from .groups import directional_table, gradient_agreement, group_metrics_json

        group_block = group_metrics_json(inputs.test, pred_arr, inputs.task)
        group_block["_gradient_agreement"] = gradient_agreement(
            directional_table(inputs.test, pred_arr, inputs.task)
        )

    report = {
        "model_name": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "task": {
            "name": inputs.task.name,
            "class_names": list(inputs.task.class_names),
            "label_map": dict(inputs.task.label_map),
            "n_classes": inputs.task.n_classes,
            "definition": inputs.task.description,
            "ties_dropped": inputs.task.drop_ties,
        },
        "split": asdict(DEFAULT_SPLIT),
        "params": getattr(detector, "params", dict)(),
        "notes": notes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "cost": {
            "train_seconds": train_seconds,
            "val_tune_seconds": val_seconds,
            "inference_seconds_per_1000": (
                infer_seconds / len(test_texts) * 1000 if test_texts else 0.0
            ),
            "test_inference_seconds": infer_seconds,
            "hatecheck_inference_seconds": hc_seconds,
            **(getattr(detector, "cost_metadata", dict)()),
        },
        "headline": classification_metrics(test_true, test_pred, names),
        "validation": classification_metrics(
            inputs.val["label"].tolist(), val_pred, names
        ),
        "slices": {"islam": slice_islam, "non_islam": slice_rest},
        "hatecheck_muslim": hc_report,
        # promoted to the top level
        "key_metrics": key_metrics(slice_islam, slice_rest, hc_report, inputs.task),
        "target_groups": group_block,
        "confidence_intervals": {
            "headline_macro_f1": bootstrap_ci(
                test_true, test_pred, "macro_f1", n_bootstrap, seed=seed
            ),
            "headline_accuracy": bootstrap_ci(
                test_true, test_pred, "accuracy", n_bootstrap, seed=seed
            ),
            "islam_macro_f1": bootstrap_ci(
                true_arr[is_slice], pred_arr[is_slice], "macro_f1",
                n_bootstrap, seed=seed,
            ),
            **{
                f"headline_recall_{name}": bootstrap_ci(
                    test_true, test_pred, f"recall_{i}", n_bootstrap, seed=seed
                )
                for i, name in enumerate(names)
            },
        },
        # kept for later McNemar
        "predictions": {
            "post_ids": inputs.test["post_id"].tolist(),
            "y_true": [int(v) for v in test_true],
            "y_pred": [int(v) for v in test_pred],
        },
        "hatecheck_predictions": {
            "case_ids": inputs.hatecheck["case_id"].astype(str).tolist(),
            "y_true": inputs.hatecheck["label"].astype(int).tolist(),
            "y_pred": [int(v) for v in hc_pred],
        },
    }

    if write:
        path = config.results_path(name, runs=to_runs_dir)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["_path"] = str(path)

    return report


# multi-seed aggregation

# never average these keys
_NON_NUMERIC_KEYS = frozenset(
    {"predictions", "hatecheck_predictions", "params", "environment",
     "task", "split", "notes", "model_name", "timestamp_utc", "_path"}
)


def _walk_numeric(reports: list[dict], reduce_fn) -> dict:
    def _combine(values: list):
        first = values[0]
        if isinstance(first, dict):
            return {
                k: _combine([v[k] for v in values if k in v])
                for k in first
                if k not in _NON_NUMERIC_KEYS
            }
        if isinstance(first, bool) or first is None or isinstance(first, str):
            return first
        if isinstance(first, (int, float)):
            return float(reduce_fn([float(v) for v in values if v is not None]))
        return first  # lists (e.g. confusion matrices) are left alone

    return _combine(reports)


def aggregate_reports(
    reports: list[dict],
    name: str,
    *,
    notes: str = "",
    write: bool = True,
) -> dict:
    if not reports:
        raise ValueError("aggregate_reports needs at least one report")

    seeds = [r["seed"] for r in reports]
    mean = _walk_numeric(reports, lambda vs: sum(vs) / len(vs))
    std = _walk_numeric(
        reports,
        lambda vs: float(np.std(vs, ddof=1)) if len(vs) > 1 else 0.0,
    )

    aggregated = dict(mean)
    first = reports[0]
    aggregated.update({
        "model_name": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seeds[0],
        "task": first["task"],
        "split": first["split"],
        "environment": first["environment"],
        "params": first["params"],
        "notes": notes,
        "predictions": first["predictions"],
        "hatecheck_predictions": first["hatecheck_predictions"],
        "runs": {
            "n_seeds": len(reports),
            "seeds": seeds,
            "predictions_from_seed": seeds[0],
            "std": std,
            "per_seed": [
                {
                    "seed": r["seed"],
                    "accuracy": r["headline"]["accuracy"],
                    "macro_f1": r["headline"]["macro_f1"],
                    # binary-only; absent under the three-class task
                    "over_flag_gap": r["key_metrics"].get("over_flag_gap"),
                    "macro_f1_gap": r["key_metrics"]["macro_f1_gap"],
                    "hatecheck_recall": r["hatecheck_muslim"]["recall_on_hateful"],
                    "hatecheck_false_alarm":
                        r["hatecheck_muslim"]["false_alarm_on_non_hateful"],
                    "key_functionalities":
                        r["key_metrics"]["hatecheck_key_functionalities"],
                    "train_seconds": r["cost"]["train_seconds"],
                }
                for r in reports
            ],
        },
    })

    if write:
        path = config.results_path(name)
        path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")
        aggregated["_path"] = str(path)

    return aggregated


def print_aggregate(report: dict) -> None:
    runs = report["runs"]
    std = runs["std"]
    km = report["key_metrics"]

    print(f"\n{'=' * 68}")
    print(f"{report['model_name']}  --  {runs['n_seeds']} runs, seeds {runs['seeds']}")
    print("=" * 68)
    print(f"\n{'seed':>6s} {'acc':>8s} {'macroF1':>8s} "
          f"{'F1 gap':>8s} {'HC rec':>8s} {'HC FA':>8s} {'train s':>8s}")
    for row in runs["per_seed"]:
        print(
            f"{row['seed']:6d} {row['accuracy']:8.4f} {row['macro_f1']:8.4f} "
            f"{row['macro_f1_gap']:+8.4f} "
            f"{row['hatecheck_recall']:8.4f} {row['hatecheck_false_alarm']:8.4f} "
            f"{row['train_seconds']:8.1f}"
        )
    print(
        f"\n  accuracy       {report['headline']['accuracy']:.4f} "
        f"± {std['headline']['accuracy']:.4f}"
    )
    print(
        f"  macro-F1       {report['headline']['macro_f1']:.4f} "
        f"± {std['headline']['macro_f1']:.4f}"
    )
    if km.get("over_flag_gap") is not None:
        print(
            f"  over-flag gap  {km['over_flag_gap']:+.4f} "
            f"± {std['key_metrics']['over_flag_gap']:.4f}"
        )
    print(
        f"  macro-F1 gap   {km['macro_f1_gap']:+.4f} "
        f"± {std['key_metrics']['macro_f1_gap']:.4f}"
    )
    print("\n  HateCheck key functionalities (mean ± std):")
    for func, value in km["hatecheck_key_functionalities"].items():
        s = std["key_metrics"]["hatecheck_key_functionalities"][func]
        print(f"    {func:20s} {value:.4f} ± {s:.4f}")


# console output


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "   n/a"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value:+.1f}%" if signed else f"{100 * value:.1f}%"


def _print_confusion(cm: dict, indent: int = 2) -> None:
    labels = cm["labels"]
    pad = " " * indent
    width = max(11, max(len(name) for name in labels) + 1)
    header = pad + f"{'truth\\pred':<{width}}" + "".join(
        f"{name:>{width}}" for name in labels
    )
    print(header)
    for name, row in zip(labels, cm["matrix"]):
        print(pad + f"{name:<{width}}" + "".join(f"{v:>{width}d}" for v in row))


def print_report(report: dict, top_functionalities: int = 8) -> None:
    head = report["headline"]
    isl = report["slices"]["islam"]
    hc = report["hatecheck_muslim"]

    print(f"\n{'=' * 68}")
    print(f"{report['model_name']}   (seed {report['seed']}, task {report['task']['name']})")
    print("=" * 68)

    print(f"\nHateXplain test (n={head['n']})")
    print(f"  accuracy   {head['accuracy']:.4f}")
    print(f"  macro-F1   {head['macro_f1']:.4f}")
    for cls, m in head["per_class"].items():
        print(
            f"  {cls:11s}  P {m['precision']:.4f}  R {m['recall']:.4f}  "
            f"F1 {m['f1']:.4f}  (n={m['support']})"
        )
    _print_confusion(head["confusion_matrix"])
    if "over_flag_rate" in head:
        print(f"  over-flag {head['over_flag_rate']:.4f}   "
              f"under-detect {head['under_detect_rate']:.4f}")

    groups = report.get("target_groups") or {}
    if groups:
        agreement = groups.get("_gradient_agreement", {})
        print("\nSEVERITY BY TARGET GROUP  (model hate-share vs human)"
              "   <- headline disaggregation")
        print(f"  {'group':12s} {'human':>8s} {'model':>8s} {'shift':>8s}"
              f" {'escal':>7s} {'de-esc':>7s}   n")
        for group, m in groups.items():
            if group.startswith("_"):
                continue
            human = m["human_hate_share"]["point"]
            model = m["model_hate_share"]["point"]
            shift = None if human is None or model is None else model - human
            mark = "" if m["human_hate_share"]["reliable"] else "  (small n)"
            print(
                f"  {group:12s} {_pct(human):>8s} {_pct(model):>8s} "
                f"{_pct(shift, signed=True):>8s} "
                f"{_pct(m['escalation_rate']['point']):>7s} "
                f"{_pct(m['de_escalation_rate']['point']):>7s}"
                f"   {m['human_hate_share']['n']}{mark}"
            )
        if agreement.get("spearman_rho") is not None:
            print(
                f"\n  Spearman rho vs human gradient: "
                f"{agreement['spearman_rho']:+.3f} "
                f"(p={agreement['p_value']:.4f}, {agreement['n_groups']} groups)"
            )
            print(f"  mean shift {agreement['mean_shift']:+.3f}")

    funcs = sorted(
        hc["per_functionality"].items(), key=lambda kv: kv[1]["pass_rate"]
    )
    print(f"\n  weakest {top_functionalities} functionalities:")
    for func, m in funcs[:top_functionalities]:
        print(
            f"    {func:28s} {m['gold']:11s} pass {m['pass_rate']:.2f} "
            f"flag {m['flag_rate']:.2f}  (n={m['n']})"
        )

    print(f"\nHateCheck, Muslim-targeted (n={hc['n_cases']})")
    print(f"  recall on {hc['n_hateful']} hateful cases      {hc['recall_on_hateful']:.4f}")
    print(f"  false alarm on {hc['n_non_hateful']} non-hateful    {hc['false_alarm_on_non_hateful']:.4f}")

    if "lenient" in hc:
        print(f"  lenient recall (hate or offensive)  "
              f"{hc['lenient']['recall_on_hateful']:.4f}"
              f"   downgraded {hc['downgraded_share_of_hateful']:+.4f}")

    # the promoted metrics
    km = report["key_metrics"]
    print("\nKEY METRICS")
    if "over_flag_gap" in km:
        print(
            f"  over-flag   islam {_fmt(km['over_flag_islam'])}  "
            f"rest {_fmt(km['over_flag_rest'])}  "
            f"gap {_fmt(km['over_flag_gap'], signed=True)}"
        )
    print(
        f"  macro-F1    islam {_fmt(km['macro_f1_islam'])}  "
        f"rest {_fmt(km['macro_f1_rest'])}  "
        f"gap {_fmt(km['macro_f1_gap'], signed=True)}"
    )
    print("  HateCheck pass rate on the four divergence functionalities:")
    for func, rate in km["hatecheck_key_functionalities"].items():
        print(f"    {func:20s} {_fmt(rate)}")

    print(f"\nWorked slice: Islam-targeted vs rest (n={isl['n']})"
          "   -- one community of the ten above")
    if isl["n"] == 0:
        print("  (empty slice)")
    else:
        print(f"  accuracy   {isl['accuracy']:.4f}   macro-F1 {isl['macro_f1']:.4f}")
        for cls, m in isl["per_class"].items():
            print(f"  {cls:11s}  P {m['precision']:.4f}  R {m['recall']:.4f}")
        _print_confusion(isl["confusion_matrix"], indent=2)

    cost = report["cost"]
    print(
        f"\nCost: train {cost['train_seconds']:.2f}s, "
        f"inference {cost['inference_seconds_per_1000']:.3f}s/1000"
    )
    # None for API-hosted models
    if cost.get("model_size_bytes"):
        print(f"      size on disk {cost['model_size_bytes'] / 1e6:.2f} MB")
    if cost.get("api_cost_usd"):
        print(f"      API cost ${cost['api_cost_usd']:.4f}")


__all__ = [
    "KEY_FUNCTIONALITIES",
    "BaseDetector",
    "Detector",
    "EvalInputs",
    "aggregate_reports",
    "bootstrap_ci",
    "classification_metrics",
    "evaluate",
    "hatecheck_report",
    "key_metrics",
    "mcnemar_test",
    "over_flag_by_original_label",
    "paired_bootstrap_diff",
    "print_aggregate",
    "print_report",
]
