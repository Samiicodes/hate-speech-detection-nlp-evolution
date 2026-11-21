from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

from src import config, data
from src.groups import group_mask
from src.models.classical_ml import TfidfLinearDetector
from src.reweight import group_balance_weights

# vocabularies -- declared, not derived, so the coding is auditable

# Severity vocabulary
SEVERITY_VOCAB = {
    "dehumanisation": """
        vermin rat rats roach roaches cockroach cockroaches parasite parasites
        subhuman animal animals ape apes monkey monkeys filth scum sludge
        disease cancer infestation infest plague swine pig pigs breed breeding
        degenerate savage savages barbarian primitive
    """.split(),
    "violence_threat": """
        kill kills killed killing murder murdered die died death dead hang
        hanged lynch gas gassed gasses exterminate extermination genocide
        slaughter shoot shot burn burned drown behead execute execution
        eliminate purge cleanse
    """.split(),
    "elimination_removal": """
        deport deported deportation expel expelled remove removal round
        roundup camps camp wall banish exile repatriate ship send
    """.split(),
}

# Form vocabulary
FORM_VOCAB = {
    "sexual_gendered": """
        bitch bitches whore whores slut sluts hoe hoes cunt pussy thot
        rape raped rapist tits body kitchen sandwich feminist feminists
        femoid roastie
    """.split(),
    "racial_slur": """
        nigger niggers nigga niggas coon coons spic spics wetback beaner
        chink chinks gook gooks negro darkie jigaboo
    """.split(),
    "religious_nativist": """
        muzzie muzzies muzrat raghead ragheads towelhead jihad jihadi sharia
        mosque quran allah infidel caliphate terrorist terrorists islamist
        immigration immigrant immigrants refugee refugees invader invaders
    """.split(),
    "antisemitic": """
        kike kikes zog goy goyim shekel shekels holocaust hooknose merchant
        zionist globalist globalists rothschild
    """.split(),
    "orientation": """
        faggot faggots fag fags dyke dykes queer queers tranny trannies
        homo sodomite groomer
    """.split(),
}

WORD_PREFIX = "word__"


def _clean(feature: str) -> str:
    return feature[len(WORD_PREFIX):] if feature.startswith(WORD_PREFIX) else ""


def categorise(feature: str) -> str | None:
    parts = feature.split()
    for name, vocab in SEVERITY_VOCAB.items():
        if any(p in vocab for p in parts):
            return f"severity:{name}"
    for name, vocab in FORM_VOCAB.items():
        if any(p in vocab for p in parts):
            return f"form:{name}"
    return None


def fit_model(texts, labels, weights, seed: int) -> TfidfLinearDetector:
    det = TfidfLinearDetector(
        name="inspect", classifier="linear_svc", tune_grid=None, seed=seed,
    )
    det.fit(texts, labels, sample_weight=weights)
    return det


def hate_weights(det: TfidfLinearDetector, task) -> tuple[np.ndarray, np.ndarray]:
    names = np.asarray(det.pipeline.named_steps["features"].get_feature_names_out())
    coef = np.atleast_2d(det.pipeline.named_steps["clf"].coef_)
    return names, coef[task.index_of("hatespeech")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    task = config.DEFAULT_TASK
    corpus = data.labelled_hatexplain(task=task).sort_values(
        "post_id", kind="mergesort"
    ).reset_index(drop=True)
    texts = corpus["text"].tolist()
    labels = corpus["label"].tolist()
    weights, _ = group_balance_weights(corpus, config.ANALYSIS_GROUPS)

    print("Fitting uncorrected and corrected models on the full corpus...")
    base = fit_model(texts, labels, None, args.seed)
    corr = fit_model(texts, labels, weights, args.seed)

    names_b, w_b = hate_weights(base, task)
    names_c, w_c = hate_weights(corr, task)
    assert (names_b == names_c).all(), "feature spaces diverged"

    # which features SURVIVE or GROW
    delta = w_c - w_b
    order_corr = np.argsort(w_c)[::-1][: args.top_k]

    print(f"\n{'=' * 92}")
    print(f"TOP {args.top_k} FEATURES TOWARD 'hatespeech' AFTER CORRECTION")
    print("=" * 92)

    counts: Counter = Counter()
    rows = []
    for idx in order_corr:
        token = _clean(str(names_b[idx]))
        if not token:
            continue
        cat = categorise(token)
        counts[cat or "uncategorised"] += 1
        rows.append({
            "token": token,
            "weight_corrected": float(w_c[idx]),
            "weight_baseline": float(w_b[idx]),
            "delta": float(delta[idx]),
            "category": cat or "uncategorised",
        })

    df = pd.DataFrame(rows)
    n_word = len(df)
    severity = sum(v for k, v in counts.items() if k.startswith("severity:"))
    form = sum(v for k, v in counts.items() if k.startswith("form:"))
    categorised = severity + form
    print(f"  word features in the top {args.top_k}: {n_word} "
          f"({int(df['token'].str.contains(' ').sum())} bigrams)")
    print(f"  severity vocabulary : {severity:4d}  ({100 * severity / n_word:.1f}%)")
    print(f"  form vocabulary     : {form:4d}  ({100 * form / n_word:.1f}%)")
    print(f"  uncategorised       : {counts['uncategorised']:4d}  "
          f"({100 * counts['uncategorised'] / n_word:.1f}%)")
    if categorised:
        print(f"\n  of the {categorised} categorised features: "
              f"{100 * severity / categorised:.1f}% severity, "
              f"{100 * form / categorised:.1f}% form")
    print("\n  by sub-category:")
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if cat == "uncategorised":
            continue
        toks = df.loc[df["category"] == cat, "token"].head(10).tolist()
        print(f"    {cat:28s} {n:4d}   {', '.join(toks)}")

    # features that GAINED most under
    print(f"\n{'=' * 92}")
    print("FEATURES THAT GAINED MOST WEIGHT UNDER CORRECTION")
    print("(what the model leans on once the label gradient is removed)")
    print("=" * 92)
    gained = df.nlargest(40, "delta")
    gain_counts = Counter(gained["category"])
    g_sev = sum(v for k, v in gain_counts.items() if k.startswith("severity:"))
    g_form = sum(v for k, v in gain_counts.items() if k.startswith("form:"))
    print(f"  of the 40 biggest gainers: {g_sev} severity, {g_form} form, "
          f"{gain_counts['uncategorised']} uncategorised")
    for _, r in gained.head(25).iterrows():
        print(f"    {r['token']:20s} {r['weight_baseline']:+.3f} -> "
              f"{r['weight_corrected']:+.3f}  ({r['delta']:+.3f})  "
              f"{r['category']}")

    # per-group: which vocabulary carries each
    print(f"\n{'=' * 92}")
    print("VOCABULARY MIX IN EACH COMMUNITY'S HOSTILE POSTS")
    print("=" * 92)
    hostile = corpus[corpus["majority_label"].isin(config.HOSTILE_LABELS)]
    token_re = re.compile(r"[a-z']+")
    all_sev = {t for v in SEVERITY_VOCAB.values() for t in v}
    all_form = {t for v in FORM_VOCAB.values() for t in v}

    grows = []
    for group in config.ANALYSIS_GROUPS:
        sub = hostile[group_mask(hostile, group)]
        toks = sub["text"].map(lambda t: set(token_re.findall(t.lower())))
        grows.append({
            "group": group,
            "n": len(sub),
            "severity_rate": float(toks.map(lambda s: bool(s & all_sev)).mean()),
            "form_rate": float(toks.map(lambda s: bool(s & all_form)).mean()),
        })
    gdf = pd.DataFrame(grows)

    human = {
        g: float(
            (hostile[group_mask(hostile, g)]["majority_label"] == "hatespeech").mean()
        )
        for g in config.ANALYSIS_GROUPS
    }
    gdf["human_hate_share"] = gdf["group"].map(human)
    gdf = gdf.sort_values("human_hate_share", ascending=False)
    show = gdf.copy()
    for col in ("severity_rate", "form_rate", "human_hate_share"):
        show[col] = show[col].map(lambda v: f"{100 * v:.1f}")
    print(show.to_string(index=False))

    from scipy import stats

    rho_sev = float(stats.spearmanr(gdf["human_hate_share"],
                                    gdf["severity_rate"]).statistic)
    p_sev = float(stats.spearmanr(gdf["human_hate_share"],
                                  gdf["severity_rate"]).pvalue)
    rho_form = float(stats.spearmanr(gdf["human_hate_share"],
                                     gdf["form_rate"]).statistic)
    print(f"\n  Spearman(hate-share, severity-vocabulary rate) {rho_sev:+.3f} "
          f"(p={p_sev:.3f})")
    print(f"  Spearman(hate-share, form-vocabulary rate)     {rho_form:+.3f}")
    print("\n  strong positive -> content really is more severe")
    print("  weak -> register, not severity")

    payload = {
        "top_k": args.top_k,
        "category_counts": dict(counts),
        "severity_fraction": severity / n_word,
        "form_fraction": form / n_word,
        "biggest_gainers": gained.head(40).to_dict(orient="records"),
        "per_group_vocabulary": gdf.to_dict(orient="records"),
        "spearman_severity": rho_sev,
        "spearman_severity_p": p_sev,
        "spearman_form": rho_form,
        "vocabularies": {
            "severity": {k: v for k, v in SEVERITY_VOCAB.items()},
            "form": {k: v for k, v in FORM_VOCAB.items()},
        },
    }
    path = config.RESULTS_DIR / "corrected_feature_analysis.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    df.to_csv(config.RESULTS_DIR / "corrected_features.csv", index=False)
    print(f"\nWritten: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
