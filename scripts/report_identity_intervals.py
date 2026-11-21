from __future__ import annotations

import argparse
import sys

import pandas as pd

from src import config
from src.groups import wilson

PLAIN = ("ident_neutral_nh", "ident_pos_nh")
ORDER = ["rule_based_lexicon", "naive_bayes", "tfidf_linear_svc", "roberta_base",
         "llm_zero_shot_haiku_corpus", "llm_few_shot_haiku_corpus"]


def load(mapping: str) -> pd.DataFrame:
    df = pd.read_csv(config.RESULTS_DIR / "hatecheck_all_identities.csv")
    df = df[df["functionality"].isin(PLAIN) & (df["identity"] != "all")]
    col = f"flag_rate_{mapping}"
    df = df.assign(flagged=(df[col] * df["n"]).round().astype(int))
    return (df.groupby(["system", "identity"], observed=True)[["flagged", "n"]]
            .sum().reset_index())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapping", choices=["strict", "lenient"], default="strict")
    args = ap.parse_args()

    tab = load(args.mapping)
    rows = []
    for _, r in tab.iterrows():
        ci = wilson(int(r["flagged"]), int(r["n"]))
        rows.append({"system": r["system"], "identity": r["identity"],
                     "n": int(r["n"]), "flagged": int(r["flagged"]),
                     "rate": ci.point, "lo": ci.lo, "hi": ci.hi})
    out = pd.DataFrame(rows)
    out["system"] = pd.Categorical(out["system"], ORDER, ordered=True)
    out = out.sort_values(["system", "rate"], ascending=[True, False])

    print(f"PLAIN-IDENTITY FALSE ALARM, {args.mapping} mapping")
    print("ident_neutral_nh + ident_pos_nh pooled, Wilson 95%\n")
    for system, g in out.groupby("system", observed=True):
        print(f"--- {system} ---")
        for _, r in g.iterrows():
            bar = f"[{r['lo']:.3f}, {r['hi']:.3f}]"
            print(f"   {r['identity']:16s} {r['flagged']:3d}/{r['n']:<4d} "
                  f"{r['rate']:.3f}  {bar}")
        _verdict(g)
        print()

    path = config.RESULTS_DIR / f"identity_intervals_{args.mapping}.csv"
    out.to_csv(path, index=False)
    print(f"written to {path}")
    return 0


def _verdict(g: pd.DataFrame) -> None:
    g = g.sort_values("rate", ascending=False)
    top = g.iloc[0]
    others = g.iloc[1:]
    if others.empty:
        return
    clears = others[others["hi"] < top["lo"]]
    if len(clears) == len(others):
        print(f"   => {top['identity']} clears ALL {len(others)} others "
              "(no interval overlap)")
    elif len(clears):
        overlap = sorted(set(others["identity"]) - set(clears["identity"]))
        print(f"   => {top['identity']} clears {len(clears)}/{len(others)}; "
              f"overlaps {', '.join(overlap)}")
    else:
        print(f"   => {top['identity']} overlaps every other identity "
              "-- no separation survives")


if __name__ == "__main__":
    sys.exit(main())
