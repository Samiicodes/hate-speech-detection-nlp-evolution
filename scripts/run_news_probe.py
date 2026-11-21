from __future__ import annotations

import argparse
import json
import re
import sys
import time

import numpy as np
import pandas as pd

from src import config, data
from src.groups import wilson
from src.models.rule_based import tokenize
from scripts.run_hatecheck_all import build_detectors, predictions_for

N_ARTICLES = 1_000
SEED = 42

# Same term lists as the corpus analysis in scripts/diagnose_identity_terms.py.
TERM_GROUPS: dict[str, tuple[str, ...]] = {
    "muslim/islam": ("muslim", "muslims", "islam", "islamic"),
    "black": ("black", "blacks"),
    "gay": ("gay", "gays"),
    "immigrant/refugee": ("immigrant", "immigrants", "refugee", "refugees"),
    "trans": ("trans", "transgender"),
    "woman": ("woman", "women"),
    "jew/jewish": ("jew", "jews", "jewish"),
}

PREDICTIONS = {
    "tfidf_linear_svc": "SVC flags Muslim-mentioning sentences MORE than others",
    "roberta_base": "RoBERTa shows NO Muslim elevation",
    "rule_based_lexicon": "lexicon shows NO Muslim elevation",
    "naive_bayes": "no prediction made",
    "llm_zero_shot_haiku_corpus": "no prediction made",
}

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def sentences(article: str) -> list[str]:
    article = re.sub(r"^\(CNN\)\s*", "", article)
    return [s.strip() for s in _SENT.split(article) if len(s.strip()) > 15]


def build_corpus() -> pd.DataFrame:
    import datasets

    ds = datasets.load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
    rng = np.random.default_rng(SEED)
    idx = sorted(int(i) for i in rng.choice(len(ds), size=N_ARTICLES, replace=False))
    rows = []
    for i in idx:
        art = ds[i]
        for j, s in enumerate(sentences(art["article"])):
            rows.append({"article_id": art["id"], "article_row": i,
                         "sent_idx": j, "text": s})
    df = pd.DataFrame(rows)
    toks = df["text"].map(lambda t: set(tokenize(t)))
    for group, terms in TERM_GROUPS.items():
        df[group] = toks.map(lambda s, t=set(terms): bool(s & t))
    df["any_group"] = df[list(TERM_GROUPS)].any(axis=1)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    args = ap.parse_args()

    task = config.DEFAULT_TASK
    HATE = task.index_of("hatespeech")
    OFF = task.index_of("offensive")

    df = build_corpus()
    print(f"{N_ARTICLES} articles (seed {SEED}) -> {len(df):,} sentences")
    print(f"  sentences mentioning >=1 community: {int(df['any_group'].sum()):,}")
    for g in TERM_GROUPS:
        print(f"    {g:20s} {int(df[g].sum()):6,}")

    train, val, _ = data.get_dataset()
    preds: dict[str, list[np.ndarray]] = {}
    for name, det in build_detectors(set(args.skip), args.seeds):
        t0 = time.perf_counter()
        p = predictions_for(det, train, val, df.rename(columns={"text": "test_case"}))
        preds.setdefault(name, []).append(p)
        print(f"  {det.name:24s} {time.perf_counter() - t0:7.1f}s")

    rows = []
    for name, runs in preds.items():
        for mapping, positive in (("strict", {HATE}), ("lenient", {HATE, OFF})):
            for group in list(TERM_GROUPS) + ["<no community>", "<all>"]:
                if group == "<all>":
                    mask = np.ones(len(df), bool)
                elif group == "<no community>":
                    mask = ~df["any_group"].to_numpy()
                else:
                    mask = df[group].to_numpy()
                n = int(mask.sum())
                if not n:
                    continue
                flags = [int(np.isin(r[mask], list(positive)).sum()) for r in runs]
                k = int(round(float(np.mean(flags))))
                ci = wilson(k, n)
                rows.append({"system": name, "mapping": mapping, "group": group,
                             "n": n, "flagged": k, "rate": ci.point,
                             "lo": ci.lo, "hi": ci.hi, "n_seeds": len(runs)})
    out = pd.DataFrame(rows)
    out.to_csv(config.RESULTS_DIR / "news_probe.csv", index=False)

    blob = {
        "n_articles": N_ARTICLES, "seed": SEED, "n_sentences": int(len(df)),
        "article_rows": sorted(df["article_row"].unique().tolist()),
        "term_groups": {k: list(v) for k, v in TERM_GROUPS.items()},
        "preregistered_predictions": PREDICTIONS,
        "preregistration_file": "results/preregistration_news_probe.md",
        "task": {"name": task.name, "n_classes": task.n_classes},
        "note": "Hate speech in edited news copy treated as ~zero; every flag "
                "is a presumed false positive. Assumption checked by manual "
                "inspection of 30 flagged sentences.",
    }
    (config.RESULTS_DIR / "news_probe.json").write_text(
        json.dumps(blob, indent=2), encoding="utf-8")

    # sentence-level predictions, needed for the manual inspection sample
    for name, runs in preds.items():
        df[f"pred_{name}"] = runs[0]
    df.to_parquet(config.RESULTS_DIR / "news_probe_sentences.parquet", index=False)

    print("\n" + "=" * 84)
    print("FLAG RATE BY COMMUNITY (strict), Wilson 95%")
    print("=" * 84)
    piv = out[out["mapping"] == "strict"].pivot(
        index="group", columns="system", values="rate")
    print(piv.round(4).to_string())
    print(f"\nwritten to {config.RESULTS_DIR / 'news_probe.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
