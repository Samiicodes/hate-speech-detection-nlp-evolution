"""Rerun systems 1 to 3 and check they reproduce the archived results exactly.

The lexicon, Naive Bayes and both linear classifiers are deterministic, so
their predictions on the held-out test split must match the archived
`results/<model>.json` row for row. Each archived file is restored after its
rerun, so the working tree is left as it was. Run `prepare_data` first.
"""

from __future__ import annotations

import json
import subprocess
import sys

from src import config

RUNS: dict[str, list[str]] = {
    "rule_based_lexicon": ["scripts.run_rule_based"],
    "naive_bayes": ["scripts.run_naive_bayes"],
    "tfidf_linear_svc": ["scripts.run_classical_ml"],
    "tfidf_logistic_regression": [
        "scripts.run_classical_ml",
        "--classifier",
        "logistic_regression",
    ],
}


def main() -> int:
    mismatches = 0
    for name, command in RUNS.items():
        path = config.RESULTS_DIR / f"{name}.json"
        archived_bytes = path.read_bytes()
        archived = json.loads(archived_bytes)
        try:
            subprocess.run(
                [sys.executable, "-m", *command],
                cwd=config.PROJECT_ROOT,
                check=True,
                capture_output=True,
            )
            rerun = json.loads(path.read_text(encoding="utf-8"))
        finally:
            path.write_bytes(archived_bytes)
        identical = rerun["predictions"] == archived["predictions"]
        mismatches += not identical
        print(
            f"{name:28s} macro-F1 archived {archived['headline']['macro_f1']:.4f}"
            f"  rerun {rerun['headline']['macro_f1']:.4f}"
            f"  predictions {'identical' if identical else 'DIFFERENT'}"
        )
    if mismatches:
        print(f"\n{mismatches} system(s) did not reproduce the archived results.")
        return 1
    print("\nAll four systems reproduce the archived results exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
