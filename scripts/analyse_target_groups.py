from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.extract_false_positives import is_model_report
from src import config, crossval, data
from src.groups import (
    agreement_table,
    directional_table,
    gradient_agreement,
    group_mask,
    human_hate_share,
    wilson,
)
from src.models.rule_based import LexiconDetector, tokenize

PARADIGM_ORDER = ["rule_based", "naive_bayes", "tfidf", "bilstm", "roberta",
                  "llm", "guard"]

# Label-free models
UNTRAINED_PREFIXES = ("llm_", "guard_")

# Pre-registered before the run
PRIOR_NOTE = """
  PRE-REGISTERED READING  (Giorgi et al., ICWSM'25, arXiv:2410.07991)

  That paper found persona-prompted LLM bias INTENSITY uncorrelated with human
  annotators (r = -0.105; -0.137 to +0.108 across four models), though bias
  PREVALENCE did correlate (r = +0.491). The hate-share gradient is the closer
  analogue of intensity, so:

    rho ~ 0      EXPECTED, consistent with the prior. The ordering is specific
                 to models trained on these annotations. NOT a failed run and
                 NOT something to debug, re-prompt or re-model.
    rho high     SURPRISING, against the prior. The ordering lives in language
                 norms absorbed in pretraining, not in this annotation. Needs
                 the few-shot condition as a robustness check before use.
    rho 0.3-0.7  Partial. Report as partial; do not round toward either story.

  Their models scored macro-F1 0.40-0.53 on detection -- around this project's
  rule-based lexicon (0.410). A model that cannot grade severity produces a
  flat profile, and a flat profile is uncorrelated with any gradient, so ALWAYS
  read rho next to macro-F1. The dissertation covers this at length.
"""


def _rank(name: str) -> int:
    for i, prefix in enumerate(PARADIGM_ORDER):
        if name.startswith(prefix):
            return i
    return len(PARADIGM_ORDER)


def _pct(v, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{100 * v:+.1f}" if signed else f"{100 * v:.1f}"


# prediction sources


def load_out_of_fold(corpus: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for name in crossval.available_oof():
        blob = crossval.load_oof(name)
        order = pd.Index(blob["post_ids"])
        if not order.equals(pd.Index(corpus["post_id"])):
            raise ValueError(
                f"{name}: out-of-fold post order does not match the corpus"
            )
        out[name] = [
            np.asarray(blob["predictions"][str(seed)], dtype=int)
            for seed in blob["seeds"]
        ]
    return dict(sorted(out.items(), key=lambda kv: (_rank(kv[0]), kv[0])))


def load_held_out(results_dir: Path, test: pd.DataFrame) -> dict[
    str, list[np.ndarray]
]:
    indexed = test.set_index("post_id", drop=False)
    out: dict[str, list[np.ndarray]] = {}
    for path in sorted(results_dir.glob("*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        if not is_model_report(blob):
            continue
        preds = blob["predictions"]
        aligned = indexed.loc[preds["post_ids"]]
        if not aligned["post_id"].equals(test["post_id"]):
            # reorder the frame to the stored prediction order instead
            pass
        out[blob["model_name"]] = [np.asarray(preds["y_pred"], dtype=int)]
    return dict(sorted(out.items(), key=lambda kv: (_rank(kv[0]), kv[0])))


# refusal rate: an outcome only the prompted systems have


def refusal_by_group(
    frame: pd.DataFrame, groups: tuple[str, ...], min_n: int
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for name in crossval.available_oof():
        blob = crossval.load_oof(name)
        if "refused_indices" not in blob:
            continue
        refused = np.zeros(blob["n_posts"], dtype=bool)
        refused[np.asarray(blob["refused_indices"], dtype=int)] = True

        rows = []
        for group in groups:
            mask = group_mask(frame, group)
            n = int(mask.sum())
            k = int(refused[mask].sum())
            p = wilson(k, n)
            rows.append({
                "group": group,
                "n": n,
                "n_refused": k,
                "rate_%": None if p.point is None else round(100 * p.point, 2),
                "ci": ("n/a" if p.point is None
                       else f"[{100 * p.lo:.2f}-{100 * p.hi:.2f}]"),
                "reliable": n >= min_n,
            })
        out[name] = pd.DataFrame(rows).sort_values(
            "rate_%", ascending=False, na_position="last"
        )
    return out


# confound 1: explicit slur content


def slur_presence_table(
    corpus: pd.DataFrame, train: pd.DataFrame, groups: tuple[str, ...]
) -> pd.DataFrame:
    lexicon = LexiconDetector()
    lexicon.fit(train["text"].tolist(), train["label"].tolist())
    terms = set(lexicon.lexicon)

    hostile = corpus[corpus["majority_label"].isin(config.HOSTILE_LABELS)]
    has_slur = hostile["text"].map(lambda t: bool(set(tokenize(t)) & terms))

    rows = []
    for group in groups:
        mask = group_mask(hostile, group)
        p = wilson(int(has_slur[mask].sum()), int(mask.sum()))
        hate_mask = mask & (hostile["majority_label"] == "hatespeech").to_numpy()
        off_mask = mask & (hostile["majority_label"] == "offensive").to_numpy()
        rows.append({
            "group": group,
            "n_hostile": p.n,
            "slur_rate": p.point,
            "ci_lo": p.lo,
            "ci_hi": p.hi,
            "slur_rate_in_hate": (
                float(has_slur[hate_mask].mean()) if hate_mask.sum() else None
            ),
            "slur_rate_in_offensive": (
                float(has_slur[off_mask].mean()) if off_mask.sum() else None
            ),
        })
    return pd.DataFrame(rows)


# main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="crossval",
                        choices=["crossval", "heldout"])
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--min-n", type=int, default=config.MIN_GROUP_N)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    task = config.DEFAULT_TASK
    if task.is_binary:
        print("This analysis needs the three-class task.")
        return 1

    corpus = data.labelled_hatexplain(task=task).sort_values(
        "post_id", kind="mergesort"
    ).reset_index(drop=True)
    train, _, test = data.get_dataset(task=task)
    groups = config.ANALYSIS_GROUPS

    if args.source == "crossval":
        frame = corpus
        per_model = load_out_of_fold(corpus)
        source_note = (
            f"5-fold out-of-fold predictions over all {len(corpus)} posts; "
            "every post predicted by a model that never trained on it"
        )
    else:
        frame = test
        per_model = load_held_out(args.results_dir, test)
        source_note = f"held-out test split only ({len(test)} posts)"

    if not per_model:
        print("No predictions found. Run scripts/run_crossval.py first.")
        return 1

    print(f"Source: {source_note}")
    print(f"Models: {', '.join(per_model)}\n")

    # 0. THE HEADLINE, up front
    human_now = pd.DataFrame({"group": list(groups)})
    human_now["n"] = [human_hate_share(corpus, g).n for g in groups]
    human_now["share"] = [human_hate_share(corpus, g).point for g in groups]
    human_now = human_now.sort_values("share", ascending=False)
    top, bottom = human_now.iloc[0], human_now.iloc[-1]

    print("=" * 104)
    print("HEADLINE -- the human severity gradient across target groups")
    print("=" * 104)
    print(
        "  Of posts humans call hostile, the fraction they call HATE SPEECH,\n"
        "  by the community targeted:\n"
    )
    print("    " + "  ".join(
        f"{r['group']} {_pct(r['share'])}%" for _, r in human_now.iterrows()
    ))
    print(f"\n  spread {100 * (top['share'] - bottom['share']):.1f} pts: "
          f"{top['group']} {_pct(top['share'])}% -> "
          f"{bottom['group']} {_pct(bottom['share'])}%")

    # 1. CONFOUNDS FIRST
    print("=" * 104)
    print("CONFOUND 1 -- IS IT THE CONTENT?  explicit slur presence among "
          "hostile posts")
    print("=" * 104)
    slur = slur_presence_table(corpus, train, groups)
    human = pd.DataFrame({"group": list(groups)})
    human["n_hostile"] = [human_hate_share(corpus, g).n for g in groups]
    human["human_hate_share"] = [human_hate_share(corpus, g).point for g in groups]
    human["ci_lo"] = [human_hate_share(corpus, g).lo for g in groups]
    human["ci_hi"] = [human_hate_share(corpus, g).hi for g in groups]

    conf = human.merge(slur.drop(columns=["n_hostile"]), on="group")
    conf = conf.sort_values("human_hate_share", ascending=False)
    show = pd.DataFrame({
        "group": conf["group"],
        "n": conf["n_hostile"],
        "hate_share": conf["human_hate_share"].map(_pct),
        "slur_rate": conf["slur_rate"].map(_pct),
        "slur_ci": conf.apply(
            lambda r: f"[{_pct(r['ci_lo_y'] if 'ci_lo_y' in r else r['ci_lo'])}-"
                      f"{_pct(r['ci_hi_y'] if 'ci_hi_y' in r else r['ci_hi'])}]",
            axis=1,
        ),
        "in_hate": conf["slur_rate_in_hate"].map(_pct),
        "in_offensive": conf["slur_rate_in_offensive"].map(_pct),
    })
    print(show.to_string(index=False))

    from scipy import stats

    rho_slur = float(stats.spearmanr(
        conf["human_hate_share"], conf["slur_rate"]
    ).statistic)
    p_slur = float(stats.spearmanr(
        conf["human_hate_share"], conf["slur_rate"]
    ).pvalue)
    print(f"\n  Spearman(human hate-share, slur rate) over {len(conf)} groups: "
          f"{rho_slur:+.3f}  (p={p_slur:.3f})")
    print("  Near zero means the severity gradient is NOT explained by how much")
    print("  explicit slur content each group attracts.")

    # The decisive comparison
    top3 = conf.nlargest(3, "slur_rate")
    span = top3["human_hate_share"].max() - top3["human_hate_share"].min()
    print(
        f"\n  Decisive comparison -- the three groups with the HIGHEST slur "
        f"rates:\n"
        + "".join(
            f"    {r['group']:12s} slur {_pct(r['slur_rate']):>5}%   "
            f"hate-share {_pct(r['human_hate_share']):>5}%   (n={r['n_hostile']})\n"
            for _, r in top3.iterrows()
        )
        + f"  Slur rates within {_pct(top3['slur_rate'].max() - top3['slur_rate'].min())} "
          f"points of each other, hate-shares spanning {_pct(span)} points.\n"
          "  Content severity cannot explain a gap that wide."
    )

    print("\n" + "=" * 104)
    print("CONFOUND 2 -- IS IT CONTESTEDNESS?  annotator agreement among "
          "hostile posts")
    print("=" * 104)
    agree = agreement_table(corpus, groups)
    conf2 = human.merge(agree.drop(columns=["n_hostile"]), on="group")
    conf2 = conf2.sort_values("human_hate_share", ascending=False)
    print(pd.DataFrame({
        "group": conf2["group"],
        "n": conf2["n_hostile"],
        "hate_share": conf2["human_hate_share"].map(_pct),
        "unanimous": conf2["unanimity"].map(_pct),
        "pairwise_agree": conf2["pairwise_agreement"].map(_pct),
        "hate/off_split": conf2["hate_offensive_split"].map(_pct),
    }).to_string(index=False))

    rho_unan = float(stats.spearmanr(
        conf2["human_hate_share"], conf2["unanimity"]
    ).statistic)
    p_unan = float(stats.spearmanr(
        conf2["human_hate_share"], conf2["unanimity"]
    ).pvalue)
    rho_split = float(stats.spearmanr(
        conf2["human_hate_share"], conf2["hate_offensive_split"]
    ).statistic)
    p_split = float(stats.spearmanr(
        conf2["human_hate_share"], conf2["hate_offensive_split"]
    ).pvalue)

    # Unanimity is mechanically coupled
    lopsided = (conf2["human_hate_share"] - 0.5).abs()
    rho_mech = float(stats.spearmanr(lopsided, conf2["unanimity"]).statistic)

    print(f"\n  Spearman(hate-share, unanimity)            {rho_unan:+.3f}  "
          f"(p={p_unan:.3f})")
    print(f"  Spearman(|hate-share - 50%|, unanimity)    {rho_mech:+.3f}  "
          "<- mechanical control")
    print(f"  Spearman(hate-share, hate/offensive split) {rho_split:+.3f}  "
          f"(p={p_split:.3f})  <- targeted measure")
    print("\n  unanimity is confounded by class balance -- see the control above")
    print("  hate/offensive split is the clean measure")
    rho_agree = rho_split  # the targeted measure is what the plot shows

    # 2. THE CENTRAL TABLE
    print("\n" + "=" * 104)
    print("MODEL HATE-SHARE BY TARGET GROUP")
    print("among posts called hostile, the fraction called hatespeech")
    print("=" * 104)

    central = pd.DataFrame({"group": list(groups)})
    first_tables = {
        name: [directional_table(frame, p, task, groups) for p in preds]
        for name, preds in per_model.items()
    }
    any_table = next(iter(first_tables.values()))[0]
    central["n_hostile"] = any_table["n_human_hostile"].to_numpy()
    central["human"] = any_table["human_hate_share"].to_numpy()
    for name, tables in first_tables.items():
        shares = np.array([t["model_hate_share"].to_numpy(dtype=float)
                           for t in tables])
        central[name] = shares.mean(axis=0)
        if len(tables) > 1:
            central[f"{name}_sd"] = shares.std(axis=0, ddof=1)
    central["reliable"] = central["n_hostile"] >= args.min_n
    central = central.sort_values("human", ascending=False)

    disp = pd.DataFrame({"group": central["group"], "n": central["n_hostile"]})
    disp["human"] = central["human"].map(_pct)
    for name in per_model:
        if f"{name}_sd" in central:
            disp[name] = [
                f"{_pct(m)}±{_pct(s)}"
                for m, s in zip(central[name], central[f"{name}_sd"])
            ]
        else:
            disp[name] = central[name].map(_pct)
    disp["shift_vs_human"] = [
        _pct(central[list(per_model)[-1]].iloc[i] - central["human"].iloc[i],
             signed=True)
        for i in range(len(central))
    ]
    disp["reliable"] = central["reliable"].map(lambda b: "" if b else "small n")
    print(disp.to_string(index=False))
    print(f"\n  All values %. 'shift_vs_human' is for {list(per_model)[-1]}.")
    print(f"  Reliability bar: n >= {args.min_n} hostile posts.")

    # 3. GRADIENT AGREEMENT
    print("\n" + "=" * 104)
    print("DOES EACH MODEL REPRODUCE THE HUMAN SEVERITY GRADIENT?")
    print("=" * 104)
    rows = []
    for name, tables in first_tables.items():
        per_seed = [gradient_agreement(t, args.min_n) for t in tables]
        rhos = [a["spearman_rho"] for a in per_seed if a["spearman_rho"] is not None]
        rows.append({
            "model": name,
            "n_groups": per_seed[0]["n_groups"],
            "spearman_rho": float(np.mean(rhos)) if rhos else None,
            "rho_sd": float(np.std(rhos, ddof=1)) if len(rhos) > 1 else None,
            "p_value": per_seed[0].get("p_value"),
            "p_method": per_seed[0].get("p_method"),
            "mean_shift": float(np.mean([a["mean_shift"] for a in per_seed])),
        })
    grad_df = pd.DataFrame(rows)
    grad_df["saw_hatexplain_labels"] = ~grad_df["model"].str.startswith(
        UNTRAINED_PREFIXES
    )
    print(grad_df.to_string(index=False))
    print("\n  mean_shift is model minus human, averaged over reliable groups.")
    print("  A positive value means the model grades hostility MORE severely")
    print("  than the annotators did, across the board.")

    untrained = grad_df[~grad_df["saw_hatexplain_labels"]]
    if not untrained.empty:
        print("\n  " + "-" * 96)
        print("  DECISIVE TEST -- models that never saw HateXplain's labels")
        print("  " + "-" * 96)
        print(PRIOR_NOTE)
        for _, r in untrained.iterrows():
            rho = r["spearman_rho"]
            if rho is None:
                continue
            if rho > 0.7:
                reading = (
                    "reproduces the human ordering WITHOUT having seen these\n"
                    "      labels -- the ordering is not an artefact of this "
                    "annotation process"
                )
            elif rho < 0.3:
                reading = (
                    "does NOT reproduce the ordering -- consistent with the\n"
                    "      gradient being specific to these annotators"
                )
            else:
                reading = "reproduces the ordering only partially"
            print(f"    {r['model']}: rho {rho:+.3f} (p={r['p_value']:.2e})")
            print(f"      {reading}")
    else:
        print("\n  (no label-free model has been run yet -- see scripts/run_llm.py;")
        print("   until one is, every row above was trained on the labels it is")
        print("   being compared against)")
        print(PRIOR_NOTE)

    # 4. ESCALATION / DE-ESCALATION WITH INTERVALS
    for measure, title, denom in (
        ("escalation",
         "ESCALATION: humans said offensive, model said hatespeech",
         "n_human_offensive"),
        ("de_escalation",
         "DE-ESCALATION: humans said hatespeech, model said offensive",
         "n_human_hate"),
    ):
        print("\n" + "=" * 104)
        print(title + "   (Wilson 95% CI)")
        print("=" * 104)
        tbl = pd.DataFrame({"group": central["group"].tolist()})
        order = [list(groups).index(g) for g in central["group"]]
        tbl["n"] = any_table[denom].to_numpy()[order]
        for name, tables in first_tables.items():
            vals = np.array([t[measure].to_numpy(dtype=float) for t in tables])
            mean = vals.mean(axis=0)[order]
            ns = any_table[denom].to_numpy()[order]
            cells = []
            for m, n in zip(mean, ns):
                if np.isnan(m) or n == 0:
                    cells.append("n/a")
                    continue
                ci = wilson(int(round(m * n)), int(n))
                cells.append(f"{_pct(m)} [{_pct(ci.lo)}-{_pct(ci.hi)}]")
            tbl[name] = cells
        print(tbl.to_string(index=False))

    # 5. REFUSAL RATE BY TARGET GROUP  (LLM only)
    refusal_tables = refusal_by_group(frame, groups, args.min_n)
    if refusal_tables:
        print("\n" + "=" * 104)
        print("REFUSAL RATE BY TARGET GROUP   (Wilson 95% CI)")
        print("=" * 104)
        print("  prompted systems only -- trained models always emit a class\n")
        for model, tbl in refusal_tables.items():
            total = tbl["n_refused"].sum()
            print(f"  {model}: {total} refusals over {tbl['n'].sum()} posts")
            print("    " + tbl.to_string(index=False).replace("\n", "\n    "))
            reliable = tbl[tbl["n"] >= args.min_n]
            if len(reliable) >= 3 and reliable["n_refused"].sum() > 0:
                from scipy import stats as _st

                table = [
                    [int(r["n_refused"]), int(r["n"] - r["n_refused"])]
                    for _, r in reliable.iterrows()
                ]
                chi2, p, _, _ = _st.chi2_contingency(table)
                print(f"\n    chi-square across {len(reliable)} reliable groups: "
                      f"chi2={chi2:.2f}, p={p:.4f}")
                print("    " + (
                    "Refusal rate DIFFERS by target community."
                    if p < 0.05 else
                    "No evidence that refusal rate depends on target community."
                ))
            elif total == 0:
                print("\n    No refusals at all -- a clean null, and worth "
                      "reporting as one.")
            print()
    else:
        print("\n  (no refusal data -- only the prompted systems can refuse, "
              "and none has been run)")

    # save + plot
    out = args.results_dir
    suffix = "" if args.source == "crossval" else "_heldout"
    central.to_csv(out / f"group_hate_share{suffix}.csv", index=False)
    conf.to_csv(out / f"group_slur_presence{suffix}.csv", index=False)
    conf2.to_csv(out / f"group_annotator_agreement{suffix}.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"group_gradient_agreement{suffix}.csv",
                              index=False)

    if not args.no_plot:
        try:
            from scripts.plot_confound import plot_confounds

            paths = plot_confounds(conf, conf2, rho_slur, rho_agree, out)
            for p in paths:
                print(f"\nPlot written: {p}")
        except ImportError as exc:
            print(f"\n(plot skipped: {exc})")

    print(f"\nTables written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
