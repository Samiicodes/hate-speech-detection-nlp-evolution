from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from scripts.run_hatecheck_all import per_identity_reports, _write_tables
from scripts.run_llm import estimate_cost
from src import config, data
from src.models.llm import ADAPTIVE_THINKING, LLMDetector, PromptTemplate, ResponseCache

PROMPTS = ("zero_shot_3class", "few_shot_3class")

# Matches the corpus-run naming
RESULT_NAME = {
    "zero_shot_3class": "llm_zero_shot_haiku_corpus",
    "few_shot_3class": "llm_few_shot_haiku_corpus",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", nargs="*", default=list(PROMPTS), choices=PROMPTS)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-batch", action="store_true")
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    task = config.DEFAULT_TASK
    hc = data.load_hatecheck().reset_index(drop=True)
    texts = hc["test_case"].tolist()
    batch = not args.no_batch
    cache = ResponseCache()

    print(f"HateCheck: {len(hc)} cases, {hc['target_ident'].nunique()} identities")
    print(f"cache holds {len(cache)} responses (hits are free)\n")

    total = 0.0
    for prompt in args.prompt:
        template = PromptTemplate.load(prompt)
        thinking = args.model.startswith(ADAPTIVE_THINKING)
        est = estimate_cost(template, texts, args.model, thinking, batch=batch)
        total += est["total_usd"]
        print(f"  {prompt:18s} {est['n_calls']} calls  ~${est['total_usd']:.2f}"
              f"  (batch={'yes' if batch else 'no'})")
    print(f"  {'TOTAL':18s} ~${total:.2f}  -- before cache hits\n")

    if args.dry_run:
        print("No calls made. Drop --dry-run to execute.")
        return 0

    blob_path = config.RESULTS_DIR / "hatecheck_all_identities.json"
    blob = json.loads(blob_path.read_text(encoding="utf-8"))

    for prompt in args.prompt:
        template = PromptTemplate.load(prompt)
        name = RESULT_NAME[prompt]
        detector = LLMDetector(
            name=name,
            prompt=prompt,
            provider="anthropic",
            provider_kwargs={
                "model": args.model,
                "temperature": 0.0,
                "effort": "low",
                "max_tokens": 1024,
                "thinking": args.model.startswith(ADAPTIVE_THINKING),
                "prompt_cache": False,
            },
            task=task,
            batch=batch,
            max_workers=args.max_workers,
            use_cache=True,
        )
        print(f"classifying {len(texts)} cases with {prompt}...")
        t0 = time.perf_counter()
        preds = np.asarray(detector.predict(texts), dtype=int)
        print(f"  {(time.perf_counter() - t0) / 60:.1f} min")

        p = detector.params()
        print(f"  refusals {p['n_refused']}  unparseable {p['n_unparseable']}")

        report = per_identity_reports(hc, preds)
        report["n_seeds"] = 1
        report["n_refused"] = p["n_refused"]
        report["n_unparseable"] = p["n_unparseable"]
        blob["systems"][name] = report
        blob.setdefault("seeds_per_system", {})[name] = 1

    blob["note"] = (
        blob.get("note", "")
        + " LLM conditions added by scripts.run_llm_hatecheck; not refitted "
          "(the model never trains), one pass per prompt condition."
    )
    blob_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"\nwritten to {blob_path}")
    _write_tables(blob, hc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
