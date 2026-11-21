from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from src import config, data
from src.harness import hatecheck_report
from src.models.naive_bayes import NaiveBayesDetector
from src.models.rule_based import LexiconDetector
from src.models.classical_ml import TfidfLinearDetector

# The four divergence functionalities
DIVERGENCE_FUNCS = (
    "ident_neutral_nh",
    "ident_pos_nh",
    "counter_quote_nh",
    "counter_ref_nh",
)

# Obfuscation functionalities
OBFUSCATION_FUNCS = (
    "spell_space_add_h",
    "spell_char_swap_h",
    "spell_char_del_h",
    "spell_leet_h",
    "spell_space_del_h",
)


def build_detectors(skip: set[str], seeds: list[int]) -> list:
    out = []
    if "rule_based_lexicon" not in skip:
        out.append(("rule_based_lexicon", LexiconDetector(name="rule_based_lexicon")))
    if "naive_bayes" not in skip:
        out.append(("naive_bayes", NaiveBayesDetector(name="naive_bayes")))
    if "tfidf_linear_svc" not in skip:
        out.append((
            "tfidf_linear_svc",
            TfidfLinearDetector(name="tfidf_linear_svc", classifier="linear_svc"),
        ))
    if "roberta_base" not in skip:
        from src.models.transformer import TransformerDetector

        for seed in seeds:
            out.append((
                "roberta_base",
                TransformerDetector(name=f"roberta_base_seed{seed}", seed=seed),
            ))
    return out


def predictions_for(detector, train, val, hatecheck) -> np.ndarray:
    detector.fit(train["text"].tolist(), train["label"].tolist())
    if hasattr(detector, "tune"):
        detector.tune(val["text"].tolist(), val["label"].tolist())
    return np.asarray(detector.predict(hatecheck["test_case"].tolist()), dtype=int)


def per_identity_reports(hc: pd.DataFrame, preds: np.ndarray) -> dict:
    out = {"all": hatecheck_report(hc, preds)}
    for ident, sub in hc.groupby("target_ident"):
        idx = sub.index.to_numpy()
        out[str(ident)] = hatecheck_report(
            sub.reset_index(drop=True), preds[idx]
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    args = parser.parse_args()

    train, val, test = data.get_dataset()
    hc = data.load_hatecheck().reset_index(drop=True)
    print(f"HateCheck: {len(hc)} cases, "
          f"{hc['target_ident'].nunique()} identities, "
          f"{hc['functionality'].nunique()} functionalities")
    print(f"  cases with no target identity: {hc['target_ident'].isna().sum()}")

    detectors = build_detectors(set(args.skip), args.seeds)
    by_system: dict[str, list[np.ndarray]] = {}
    for name, det in detectors:
        t0 = time.time()
        preds = predictions_for(det, train, val, hc)
        by_system.setdefault(name, []).append(preds)
        print(f"  {det.name:24s} {time.time() - t0:7.1f}s")

    blob = {
        "n_cases": int(len(hc)),
        "identities": sorted(hc["target_ident"].dropna().unique().tolist()),
        "functionalities": sorted(hc["functionality"].unique().tolist()),
        "task": {
            "name": config.DEFAULT_TASK.name,
            "n_classes": config.DEFAULT_TASK.n_classes,
        },
        "seeds_per_system": {k: len(v) for k, v in by_system.items()},
        "note": (
            "Refitted on the standard training split; HateCheck is evaluation "
            "only. RoBERTa entries are the mean over seeds, matching how "
            "results/roberta_base.json aggregates its Muslim-slice HateCheck."
        ),
        "systems": {},
    }

    blob["per_seed"] = {}
    for name, runs in by_system.items():
        reports = [per_identity_reports(hc, p) for p in runs]
        blob["systems"][name] = _mean_of(reports)
        blob["systems"][name]["n_seeds"] = len(runs)
        if len(runs) > 1:
            # Keep the individual seeds
            blob["per_seed"][name] = [
                {"seed": s, "headline": {
                    ident: {k: r[ident][k] for k in
                            ("recall_on_hateful", "false_alarm_on_non_hateful")}
                    for ident in r}}
                for s, r in zip(args.seeds, reports)
            ]

    check = _check_muslim_slice_reproduces(blob)
    blob["muslim_slice_reproduction"] = check
    print()
    for line in check["lines"]:
        print("   ", line)

    # Write first, judge second
    out = config.RESULTS_DIR / "hatecheck_all_identities.json"
    out.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")
    _write_tables(blob, hc)

    if check["deterministic_failures"]:
        print("\nA DETERMINISTIC SYSTEM FAILED TO REPRODUCE -- results above are "
              "not comparable to results/*.json. Investigate before using them.")
        return 1
    return 0

    _write_tables(blob, hc)
    return 0


# Deterministic systems only
EXACTLY_REPRODUCIBLE = ("rule_based_lexicon", "naive_bayes", "tfidf_linear_svc")


def _check_muslim_slice_reproduces(blob: dict, tol: float = 1e-9) -> dict:
    checks = (
        ("recall strict", ("recall_on_hateful",)),
        ("false alarm strict", ("false_alarm_on_non_hateful",)),
        ("recall lenient", ("lenient", "recall_on_hateful")),
        ("false alarm lenient", ("lenient", "false_alarm_on_non_hateful")),
    )
    lines, deltas, failures = [], {}, []
    for system, report in blob["systems"].items():
        path = config.RESULTS_DIR / f"{system}.json"
        if not path.exists():
            continue
        stored = json.loads(path.read_text(encoding="utf-8")).get("hatecheck_muslim")
        if stored is None or "Muslims" not in report:
            continue
        exact = system in EXACTLY_REPRODUCIBLE
        worst = 0.0
        for label, keys in checks:
            a, b = stored, report["Muslims"]
            for k in keys:
                a = a.get(k) if isinstance(a, dict) else None
                b = b.get(k) if isinstance(b, dict) else None
            if a is None or b is None:
                continue
            worst = max(worst, abs(a - b))
            if exact and abs(a - b) > tol:
                failures.append(f"{system} {label}: stored {a:.4f} refit {b:.4f}")
            elif not exact:
                deltas[f"{system}.{label}"] = {"stored": a, "refit": b,
                                               "delta": b - a}
        if exact:
            lines.append(
                f"{system:22s} reproduces the stored Muslim slice exactly"
                if worst <= tol else
                f"{system:22s} DOES NOT REPRODUCE (worst |delta| {worst:.4f})"
            )
        else:
            lines.append(
                f"{system:22s} stochastic (deterministic=False); rerun moves "
                f"the Muslim slice by up to {worst:.4f}"
            )
    return {"lines": lines, "stochastic_deltas": deltas,
            "deterministic_failures": failures}


def _mean_of(reports: list[dict]):
    if len(reports) == 1:
        return reports[0]
    first = reports[0]
    if isinstance(first, dict):
        return {k: _mean_of([r[k] for r in reports]) for k in first}
    if isinstance(first, (int, float)):
        return float(np.mean(reports))
    return first


def _write_tables(blob: dict, hc: pd.DataFrame) -> None:
    rows = []
    for system, report in blob["systems"].items():
        for ident, r in report.items():
            if not isinstance(r, dict) or "per_functionality" not in r:
                continue
            for func, f in r["per_functionality"].items():
                rows.append({
                    "system": system, "identity": ident, "functionality": func,
                    "gold": f["gold"], "n": f["n"],
                    "flag_rate_strict": f["flag_rate"],
                    "flag_rate_lenient": f["flag_rate_lenient"],
                    "pass_rate_strict": f["pass_rate"],
                })
    df = pd.DataFrame(rows)
    path = config.RESULTS_DIR / "hatecheck_all_identities.csv"
    df.to_csv(path, index=False)
    print(f"written to {path}   ({len(df)} rows)")


if __name__ == "__main__":
    sys.exit(main())
