from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from src import config, data
from scripts.extract_false_positives import load_predictions, mask_slurs

CATEGORY_ORDER = [
    "slur_dehumanisation",
    "nativist_political",
    "doctrinal_attack",
    "other_group_primary",
    "ingroup_intraminority",
    "counter_speech",
]

# Silencing categories
CONCERNING = {"counter_speech", "ingroup_intraminority"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-per-category", type=int, default=3)
    parser.add_argument(
        "--coding", type=Path,
        default=config.RESULTS_DIR / "error_analysis_coding.json",
    )
    args = parser.parse_args()

    coding = json.loads(args.coding.read_text(encoding="utf-8"))
    _, _, test = data.get_dataset()
    indexed = test.set_index("post_id")
    preds = load_predictions(config.RESULTS_DIR)

    roberta = coding["roberta_base"]
    svc_only = coding["tfidf_linear_svc_only"]

    pool = test[(test["label"] == 0) & test["is_islam_slice"]]
    n_pool = len(pool)

    lines: list[str] = [
        "# Error analysis: what RoBERTa's Islam-targeted false positives actually are",
        "",
        "Every non-hateful, Islam-targeted post in the HateXplain test split that",
        f"`roberta_base` (seed 42) flags as hate. **{len(roberta)} posts** out of a",
        f"pool of {n_pool} -- the full population, not a sample.",
        "",
        "Slurs are masked. Posts are quoted verbatim otherwise, including the",
        "original spelling and the absence of punctuation.",
        "",
        "## Coding scheme",
        "",
    ]
    for cat in CATEGORY_ORDER:
        lines.append(f"- **`{cat}`** — {coding['categories'][cat]}")

    # distribution
    counts = Counter(roberta.values())
    lines += ["", "## Distribution", "",
              "| category | n | % of flagged |", "|---|---|---|"]
    for cat in CATEGORY_ORDER:
        n = counts.get(cat, 0)
        lines.append(f"| `{cat}` | {n} | {100 * n / len(roberta):.1f}% |")
    concerning = sum(counts.get(c, 0) for c in CONCERNING)
    lines += [
        f"| **total** | **{len(roberta)}** | |",
        "",
        f"Posts where the model silences someone describing or experiencing "
        f"abuse rather than committing it: **{concerning} of {len(roberta)} "
        f"({100 * concerning / len(roberta):.1f}%)**.",
        "",
    ]

    # split by original label
    by_label: dict[str, Counter] = {"normal": Counter(), "offensive": Counter()}
    for post_id, cat in roberta.items():
        by_label[indexed.loc[post_id, "majority_label"]][cat] += 1
    lines += ["## By original three-class label", "",
              "| category | normal | offensive |", "|---|---|---|"]
    for cat in CATEGORY_ORDER:
        lines.append(
            f"| `{cat}` | {by_label['normal'].get(cat, 0)} | "
            f"{by_label['offensive'].get(cat, 0)} |"
        )
    lines += [
        f"| **total** | **{sum(by_label['normal'].values())}** | "
        f"**{sum(by_label['offensive'].values())}** |",
        "",
    ]

    # examples
    lines += ["## Examples", ""]
    for cat in CATEGORY_ORDER:
        ids = [p for p, c in roberta.items() if c == cat]
        if not ids:
            continue
        lines += [f"### `{cat}` ({len(ids)})", ""]
        for post_id in ids[: args.examples_per_category]:
            row = indexed.loc[post_id]
            others = ", ".join(
                f"{m}={preds[m].get(post_id)}"
                for m in sorted(preds) if m != "roberta_base"
            )
            lines += [
                f"> {mask_slurs(row['text'])}",
                "",
                f"*{post_id} · annotators: {row['majority_label']} · {others}*",
                "",
            ]

    # paradigm comparison
    svc_preds = preds.get("tfidf_linear_svc", {})
    overlap = [p for p in roberta if svc_preds.get(p) == 1]
    roberta_only = [p for p in roberta if svc_preds.get(p) != 1]

    lines += [
        "## Does the picture shift by paradigm?",
        "",
        f"The SVM flags {sum(1 for p in pool['post_id'] if svc_preds.get(p) == 1)} "
        f"posts from the same pool; RoBERTa flags {len(roberta)}. "
        f"{len(overlap)} are flagged by both.",
        "",
        "| category | both | RoBERTa only | SVM only |",
        "|---|---|---|---|",
    ]
    both_counts = Counter(roberta[p] for p in overlap)
    only_counts = Counter(roberta[p] for p in roberta_only)
    svc_counts = Counter(svc_only.values())
    for cat in CATEGORY_ORDER:
        lines.append(
            f"| `{cat}` | {both_counts.get(cat, 0)} | "
            f"{only_counts.get(cat, 0)} | {svc_counts.get(cat, 0)} |"
        )
    lines += [
        f"| **total** | **{len(overlap)}** | **{len(roberta_only)}** | "
        f"**{len(svc_only)}** |",
        "",
    ]

    svc_concerning = sum(svc_counts.get(c, 0) for c in CONCERNING)
    lines += [
        f"Of the {len(svc_only)} posts only the SVM flags, **{svc_concerning}** "
        "fall in the two concerning categories "
        f"({100 * svc_concerning / max(1, len(svc_only)):.0f}%), against "
        f"{concerning} of {len(roberta)} ({100 * concerning / len(roberta):.0f}%) "
        "across all of RoBERTa's.",
        "",
        "### Posts only the SVM flags",
        "",
    ]
    for post_id, cat in svc_only.items():
        row = indexed.loc[post_id]
        lines += [
            f"> {mask_slurs(row['text'])}",
            "",
            f"*{post_id} · annotators: {row['majority_label']} · `{cat}`*",
            "",
        ]

    out = config.RESULTS_DIR / "error_analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Pool: {n_pool} Islam-targeted non-hateful test posts")
    print(f"RoBERTa flags: {len(roberta)} | SVM flags: "
          f"{sum(1 for p in pool['post_id'] if svc_preds.get(p) == 1)} | "
          f"both: {len(overlap)}")
    print("\nCategory distribution (RoBERTa):")
    for cat in CATEGORY_ORDER:
        n = counts.get(cat, 0)
        flag = "  <-- concerning" if cat in CONCERNING else ""
        print(f"  {cat:24s} {n:3d}  {100 * n / len(roberta):5.1f}%{flag}")
    print(f"\nConcerning total: {concerning}/{len(roberta)} "
          f"({100 * concerning / len(roberta):.1f}%)")
    print(f"SVM-only concerning: {svc_concerning}/{len(svc_only)} "
          f"({100 * svc_concerning / max(1, len(svc_only)):.0f}%)")
    print(f"\nWritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
