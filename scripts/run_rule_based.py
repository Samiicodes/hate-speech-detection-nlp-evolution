from __future__ import annotations

import argparse
import sys

from src import config, data
from src.harness import EvalInputs, evaluate, print_report
from src.models.rule_based import LexiconDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--min-freq", type=int, default=40)
    parser.add_argument("--match-threshold", type=int, default=1)
    parser.add_argument("--unweighted", action="store_true",
                        help="use the plain log-odds ratio instead of the z-scored one")
    parser.add_argument("--name", default="rule_based_lexicon")
    args = parser.parse_args()

    train, val, test = data.get_dataset()
    hatecheck = data.hatecheck_muslim()

    detector = LexiconDetector(
        name=args.name,
        top_k=args.top_k,
        min_freq=args.min_freq,
        match_threshold=args.match_threshold,
        weighted=not args.unweighted,
    )

    report = evaluate(
        detector,
        EvalInputs(train=train, val=val, test=test, hatecheck=hatecheck),
        notes=(
            "Lexicon derived from the training split only. Protected-group "
            "identity terms are stoplisted by design; see lexicon_stoplist.py."
        ),
    )
    print_report(report)

    path = detector.save_lexicon()
    print(f"\nLexicon ({len(detector.lexicon)} tokens) written to {path}")
    print("Top 40:", ", ".join(detector.lexicon[:40]))
    print(f"Results written to {config.results_path(detector.name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
