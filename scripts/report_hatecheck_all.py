from __future__ import annotations

import argparse
import sys

import pandas as pd

from src import config
from scripts.run_hatecheck_all import DIVERGENCE_FUNCS, OBFUSCATION_FUNCS

SYSTEM_ORDER = ["rule_based_lexicon", "naive_bayes", "tfidf_linear_svc", "roberta_base"]


def load() -> pd.DataFrame:
    path = config.RESULTS_DIR / "hatecheck_all_identities.csv"
    df = pd.read_csv(path)
    df["system"] = pd.Categorical(df["system"], SYSTEM_ORDER, ordered=True)
    return df.sort_values(["system", "identity", "functionality"])


def _wide(df, funcs, value, decimals=3):
    sub = df[df["functionality"].isin(funcs) & (df["identity"] != "all")]
    return sub.pivot_table(index=["system", "functionality"],
                           columns="identity", values=value).round(decimals)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapping", choices=["strict", "lenient"], default="strict")
    args = ap.parse_args()
    col = f"flag_rate_{args.mapping}"

    df = load()
    idents = sorted(df.loc[df["identity"] != "all", "identity"].unique())
    print(f"HateCheck, all identities. mapping = {args.mapping}")
    print(f"identities: {', '.join(idents)}\n")

    print("=" * 78)
    print("1. THE FOUR SINGLED-OUT FUNCTIONALITIES, BY IDENTITY")
    print("   flag rate; all four are non-hateful, so lower is better")
    print("=" * 78)
    w = _wide(df, DIVERGENCE_FUNCS, col)
    print(w.to_string())

    print("\n" + "=" * 78)
    print("2. SPREAD ACROSS IDENTITIES  (max - min flag rate)")
    print("=" * 78)
    spread = w.assign(spread=w.max(axis=1) - w.min(axis=1),
                      worst=w.idxmax(axis=1), best=w.idxmin(axis=1))
    print(spread[["spread", "worst", "best"]].to_string())

    print("\n" + "=" * 78)
    print("3. PLAIN-IDENTITY FALSE ALARM, POOLED OVER THE TWO ident_* FUNCTIONS")
    print("=" * 78)
    plain = df[df["functionality"].isin(("ident_neutral_nh", "ident_pos_nh"))
               & (df["identity"] != "all")]
    pooled = (plain.assign(flagged=plain[col] * plain["n"])
              .groupby(["system", "identity"], observed=True)[["flagged", "n"]].sum())
    pooled["rate"] = pooled["flagged"] / pooled["n"]
    tab = pooled["rate"].unstack().round(3)
    tab["spread"] = (tab.max(axis=1) - tab.min(axis=1)).round(3)
    print(tab.to_string())

    print("\n" + "=" * 78)
    print("4. CHARACTER OBFUSCATION, FULL SUITE vs MUSLIM ONLY")
    print("   flag rate; all hateful, so higher is better")
    print("=" * 78)
    obf = df[df["functionality"].isin(OBFUSCATION_FUNCS)]
    full = (obf[obf["identity"] != "all"]
            .assign(f=lambda d: d[col] * d["n"])
            .groupby(["system", "functionality"], observed=True)[["f", "n"]].sum())
    full["full_suite"] = full["f"] / full["n"]
    mus = (obf[obf["identity"] == "Muslims"]
           .set_index(["system", "functionality"])[col].rename("muslims_only"))
    both = full[["full_suite", "n"]].join(mus).round(3)
    print(both.to_string())

    out = config.RESULTS_DIR / f"hatecheck_identity_spread_{args.mapping}.csv"
    tab.to_csv(out)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
