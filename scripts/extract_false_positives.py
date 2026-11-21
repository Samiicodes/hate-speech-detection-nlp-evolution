from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


from src import config, data

# Kept separate from the lexicon
SLURS = [
    "nigger", "niggers", "nigga", "niggas", "kike", "kikes", "muzzie",
    "muzzies", "muzrat", "raghead", "ragheads", "spic", "spics", "paki",
    "pakis", "negro", "chink", "chinks", "gook", "gooks", "coon", "coons",
    "faggot", "faggots", "fag", "fags", "dyke", "dykes", "queer", "queers",
    "tranny", "trannies", "retard", "retards", "retarded", "shitskin",
    "mudshark", "wetback", "beaner", "cunt", "cunts", "bitch", "bitches",
    "whore", "whores", "slut", "sluts", "hoe", "hoes",
]
_SLUR_RE = re.compile(r"\b(" + "|".join(SLURS) + r")\b", re.IGNORECASE)


def mask_slurs(text: str) -> str:
    return _SLUR_RE.sub(lambda m: m.group(0)[0] + "*" * (len(m.group(0)) - 1), text)


def is_model_report(blob: object) -> bool:
    return (
        isinstance(blob, dict)
        and "model_name" in blob
        and "predictions" in blob
        and "headline" in blob
    )


def load_predictions(results_dir: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for path in sorted(results_dir.glob("*.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        if not is_model_report(blob):
            continue
        preds = blob["predictions"]
        out[blob["model_name"]] = dict(zip(preds["post_ids"], preds["y_pred"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="roberta_base")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument(
        "--bucket", default="all", choices=["all", "normal", "offensive"],
        help="restrict to one original 3-class label",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    preds = load_predictions(args.results_dir)
    if args.model not in preds:
        print(f"No results for {args.model}. Have: {sorted(preds)}")
        return 1

    _, _, test = data.get_dataset()
    # non-hateful, Islam-targeted -- the population the analysis is about
    pool = test[(test["label"] == 0) & test["is_islam_slice"]].copy()
    if args.bucket != "all":
        pool = pool[pool["majority_label"] == args.bucket]

    pool["flagged"] = pool["post_id"].map(preds[args.model])
    flagged = pool[pool["flagged"] == 1].copy()

    for model, mapping in preds.items():
        flagged[f"pred_{model}"] = flagged["post_id"].map(mapping)

    flagged["text_masked"] = flagged["text"].map(mask_slurs)

    print(f"Islam-targeted non-hateful test posts: {len(pool)}")
    print(f"  by original label: {dict(pool['majority_label'].value_counts())}")
    print(f"Flagged by {args.model}: {len(flagged)}")
    print(f"  by original label: {dict(flagged['majority_label'].value_counts())}")

    cols = ["post_id", "majority_label", "text_masked"] + [
        f"pred_{m}" for m in sorted(preds)
    ]
    out = args.out or (config.RESULTS_DIR / f"false_positives_{args.model}.csv")
    flagged[cols].to_csv(out, index=False)
    print(f"\nWritten: {out}")

    print("\n" + "=" * 88)
    for i, row in enumerate(flagged.itertuples(), start=1):
        others = " ".join(
            f"{m.replace('tfidf_', '').replace('_lexicon', '')}="
            f"{getattr(row, f'pred_{m}')}"
            for m in sorted(preds) if m != args.model
        )
        print(f"[{i:3d}] {row.post_id}  ({row.majority_label})  {others}")
        print(f"      {row.text_masked}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
