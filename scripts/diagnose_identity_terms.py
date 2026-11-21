from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src import config, data
from src.models.naive_bayes import NaiveBayesDetector
from src.models.rule_based import LexiconDetector
from src.models.classical_ml import TfidfLinearDetector

# Identity to template unigrams
IDENT_TERMS: dict[str, tuple[str, ...]] = {
    "Muslims": ("muslim", "muslims"),
    "black people": ("black",),
    "disabled people": ("disabled",),
    "gay people": ("gay",),
    "immigrants": ("immigrant", "immigrants"),
    "trans people": ("trans",),
    "women": ("woman", "women"),
}

# HateCheck identity to corpus group
IDENT_TO_GROUP: dict[str, str | None] = {
    "Muslims": "islam",
    "black people": "african",
    "disabled people": None,
    "gay people": "homosexual",
    "immigrants": "refugee",
    "trans people": None,
    "women": "women",
}


def term_mask(texts: pd.Series, terms: tuple[str, ...]) -> np.ndarray:
    from src.models.rule_based import tokenize

    want = set(terms)
    return texts.map(lambda t: bool(set(tokenize(t)) & want)).to_numpy()


def check_one(train: pd.DataFrame) -> pd.DataFrame:
    task = config.DEFAULT_TASK
    names = task.class_names
    rows = []
    for ident, terms in IDENT_TERMS.items():
        m = term_mask(train["text"], terms)
        n = int(m.sum())
        row = {"identity": ident, "terms": "/".join(terms), "n_posts": n}
        for i, cls in enumerate(names):
            row[cls] = float((train["label"].to_numpy()[m] == i).mean()) if n else np.nan
        row["hostile"] = row[names[0]] + row[names[1]] if n else np.nan
        rows.append(row)

    base = {"identity": "ALL TRAINING POSTS", "terms": "-", "n_posts": len(train)}
    for i, cls in enumerate(names):
        base[cls] = float((train["label"] == i).mean())
    base["hostile"] = base[names[0]] + base[names[1]]
    rows.append(base)
    return pd.DataFrame(rows)


def group_slurs(train: pd.DataFrame, corpus: pd.DataFrame, min_share: float = 0.5):
    from src.groups import group_mask
    from src.models.rule_based import tokenize

    lex = LexiconDetector()
    lex.fit(train["text"].tolist(), train["label"].tolist())
    terms = set(lex.lexicon)

    tokens = train["text"].map(lambda t: set(tokenize(t)))
    ids = train["post_id"].astype(str)
    tagged = corpus.set_index(corpus["post_id"].astype(str))

    out: dict[str, list[str]] = {}
    for ident, group in IDENT_TO_GROUP.items():
        if group is None:
            out[ident] = []
            continue
        sub = tagged.loc[tagged.index.intersection(ids)]
        in_group = pd.Series(group_mask(sub, group).astype(bool), index=sub.index)
        found = []
        for term in terms:
            has = tokens.map(lambda s, t=term: t in s).to_numpy()
            if has.sum() < 5:
                continue
            share = in_group.reindex(ids[has]).fillna(False).mean()
            if share >= min_share:
                found.append(term)
        out[ident] = found
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-share", type=float, default=0.5)
    ap.parse_args()

    task = config.DEFAULT_TASK
    train, val, _ = data.get_dataset()
    corpus = data.labelled_hatexplain(task=task)

    print("=" * 78)
    print("CHECK 1 -- class composition of training posts containing each term")
    print("=" * 78)
    one = check_one(train)
    print(one.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n" + "=" * 78)
    print("CHECK 2 -- linear SVC weight toward hatespeech")
    print("=" * 78)
    svc = TfidfLinearDetector(name="tfidf_linear_svc", classifier="linear_svc")
    svc.fit(train["text"].tolist(), train["label"].tolist())
    svc.tune(val["text"].tolist(), val["label"].tolist())
    svc_w = _weights(svc, task)

    print("\n" + "=" * 78)
    print("CHECK 3 -- Naive Bayes log-probability ratio, hatespeech vs normal")
    print("=" * 78)
    nb = NaiveBayesDetector(name="naive_bayes")
    nb.fit(train["text"].tolist(), train["label"].tolist())
    nb.tune(val["text"].tolist(), val["label"].tolist())
    nb_w = _nb_ratio(nb, task)

    slurs = group_slurs(train, corpus, args_min_share=None) if False else \
        group_slurs(train, corpus)
    rows = []
    for ident, terms in IDENT_TERMS.items():
        neutral = max(terms, key=lambda t: svc_w.get(t, -np.inf))
        cand = slurs.get(ident, [])
        best_slur = max(cand, key=lambda t: svc_w.get(t, -np.inf), default=None)
        rows.append({
            "identity": ident,
            "neutral": neutral,
            "svc_w_neutral": svc_w.get(neutral),
            "svc_rank_neutral": _rank(svc_w, neutral),
            "slur": best_slur,
            "svc_w_slur": svc_w.get(best_slur) if best_slur else None,
            "svc_gap": (svc_w.get(best_slur) - svc_w.get(neutral))
                       if best_slur and neutral in svc_w else None,
            "nb_logratio_neutral": nb_w.get(neutral),
            "nb_logratio_slur": nb_w.get(best_slur) if best_slur else None,
            "n_group_slurs": len(cand),
        })
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    out = config.RESULTS_DIR / "identity_term_weights.csv"
    tab.to_csv(out, index=False)
    one.to_csv(config.RESULTS_DIR / "identity_term_composition.csv", index=False)
    print(f"\nwritten to {out}")
    return 0


def _weights(det, task) -> dict[str, float]:
    pipe = det.pipeline
    feats = pipe.named_steps["features"] if "features" in pipe.named_steps else None
    clf = pipe.steps[-1][1]
    names = np.asarray(feats.get_feature_names_out()) if feats is not None else None
    if names is None:
        raise SystemExit("could not locate the feature union on the SVC pipeline")
    hate = task.index_of("hatespeech")
    coef = clf.coef_[hate]
    word = np.array([n.startswith("word__") for n in names])
    return {n.split("__", 1)[1]: float(w)
            for n, w, is_w in zip(names, coef, word) if is_w}


def _rank(weights: dict[str, float], term: str) -> int | None:
    if term not in weights:
        return None
    order = sorted(weights, key=weights.get, reverse=True)
    return order.index(term) + 1


def _nb_ratio(det, task) -> dict[str, float]:
    pipe = det.pipeline
    vec = pipe.named_steps["counts"]
    clf = pipe.named_steps["clf"]
    names = vec.get_feature_names_out()
    h, n = task.index_of("hatespeech"), task.index_of("normal")
    diff = clf.feature_log_prob_[h] - clf.feature_log_prob_[n]
    return {str(t): float(d) for t, d in zip(names, diff)}


if __name__ == "__main__":
    sys.exit(main())
