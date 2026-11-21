from __future__ import annotations

import argparse
import sys

from src import config, data
from src.groups import available_groups, human_hate_share

# Pilot figures: group -> (hate_share_pct, ci_lo_pct, ci_hi_pct, n_hostile)
PILOT: dict[str, tuple[float, float, float, int]] = {
    "jewish":     (77.2, 75.2, 79.1, 1804),
    "african":    (75.8, 74.2, 77.3, 2965),
    "hispanic":   (75.4, 70.5, 79.7, 337),
    "arab":       (72.2, 68.8, 75.4, 699),
    "islam":      (66.0, 63.7, 68.1, 1815),
    "asian":      (54.7, 49.0, 60.3, 298),
    "homosexual": (46.6, 44.1, 49.1, 1541),
    "refugee":    (32.8, 29.0, 36.9, 533),
    "women":      (29.9, 27.5, 32.5, 1267),
    "caucasian":  (28.6, 24.2, 33.4, 371),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tol-pct", type=float, default=0.15,
                        help="allowed absolute difference in percentage points")
    parser.add_argument("--show-all-groups", action="store_true")
    args = parser.parse_args()

    df = data.labelled_hatexplain(task=config.TASK_THREE_CLASS)
    n_all = len(data.load_hatexplain(task=config.TASK_THREE_CLASS))
    print(f"Corpus: {n_all} posts, {n_all - len(df)} tied and dropped, "
          f"{len(df)} with a majority label\n")

    header = (f"{'group':12s} {'n':>6s} {'expect':>7s} {'got':>7s} "
              f"{'CI (pilot)':>15s} {'CI (here)':>15s}   status")
    print(header)
    print("-" * len(header))

    ok = True
    for group, (p_pct, lo_pct, hi_pct, n_exp) in PILOT.items():
        obs = human_hate_share(df, group)
        got_pct = 100 * obs.point if obs.point is not None else float("nan")
        n_match = obs.n == n_exp
        p_match = abs(got_pct - p_pct) <= args.tol_pct
        ci_match = (
            abs(100 * obs.lo - lo_pct) <= args.tol_pct
            and abs(100 * obs.hi - hi_pct) <= args.tol_pct
        )
        good = n_match and p_match and ci_match
        ok &= good

        status = "ok" if good else (
            "N MISMATCH" if not n_match else
            "RATE MISMATCH" if not p_match else "CI MISMATCH"
        )
        print(
            f"{group:12s} {obs.n:6d} {p_pct:6.1f}% {got_pct:6.1f}% "
            f"  [{lo_pct:4.1f}-{hi_pct:4.1f}]   [{100 * obs.lo:4.1f}-{100 * obs.hi:4.1f}]"
            f"   {status}"
        )
        if not n_match:
            print(f"{'':12s} expected n={n_exp}, got n={obs.n}")

    if args.show_all_groups:
        print("\nAll groups present in the corpus:")
        for group in available_groups(df):
            obs = human_hate_share(df, group)
            mark = "" if obs.reliable else "   (n too small for a claim)"
            print(f"  {group:16s} {obs}{mark}")

    if ok:
        print("\nHuman baseline reproduced exactly.")
        return 0
    print("\nBaseline does NOT reproduce -- group filtering differs from the pilot.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
