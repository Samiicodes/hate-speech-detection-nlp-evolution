from __future__ import annotations

import argparse
import sys

import pandas as pd

from scripts.run_llm import (
    MIN_CACHEABLE_TOKENS,
    TIERS,
    CHARS_PER_TOKEN,
    estimate_cost,
)
from src import config, data, env
from src.models.llm import ADAPTIVE_THINKING, PromptTemplate

PROMPTS = ("zero_shot_3class", "few_shot_3class")


def prefix_tokens(template: PromptTemplate, model: str) -> tuple[int, str]:
    if env.has("ANTHROPIC_API_KEY"):
        try:
            from src.models.llm import AnthropicProvider

            provider = AnthropicProvider(model=model)
            _, messages = template.render("")
            # everything except the final (variable) user turn
            shared = messages[:-1]
            n = provider.count_tokens(template.system, shared or [
                {"role": "user", "content": "x"}
            ])
            return n, "count_tokens"
        except Exception as exc:  # noqa: BLE001
            print(f"  (count_tokens unavailable: {exc}; falling back)")
    chars = len(template.system) + sum(
        len(template.user.format(text=e["text"])) + len(e["label"])
        for e in template.examples
    )
    return int(chars / CHARS_PER_TOKEN), "estimate"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    task = config.DEFAULT_TASK
    _, val, test = data.get_dataset(task=task)
    hatecheck = data.hatecheck_muslim()
    corpus = data.labelled_hatexplain(task=task)

    scopes = {
        "corpus": len(corpus),
        "heldout": len(val) + len(test) + len(hatecheck),
    }

    print("=" * 100)
    print("SYSTEM 4 COST GRID")
    print("=" * 100)
    print(f"  corpus tier   {scopes['corpus']:,} posts")
    print(f"  heldout tier  {scopes['heldout']:,} calls "
          f"(val {len(val)} + test {len(test)} + hatecheck {len(hatecheck)})")
    print(f"  batch API     50% off input and output")
    print(f"  prompt cache  needs a shared prefix >= {MIN_CACHEABLE_TOKENS} tokens")

    print("\n  shared prefix size (system message + any few-shot turns):")
    prefixes = {}
    for tier, cfg in TIERS.items():
        for prompt in PROMPTS:
            template = PromptTemplate.load(prompt)
            n, source = prefix_tokens(template, cfg["model"])
            prefixes[(tier, prompt)] = n
            ok = "CACHEABLE" if n >= MIN_CACHEABLE_TOKENS else "too short"
            print(f"    {tier:8s} {prompt:18s} {n:5d} tokens ({source})  -> {ok}")

    rows = []
    for tier, cfg in TIERS.items():
        model = cfg["model"]
        n_calls = scopes[tier]
        texts = (
            corpus["text"].tolist() if tier == "corpus"
            else val["text"].tolist() + test["text"].tolist()
            + hatecheck["test_case"].tolist()
        )
        for prompt in PROMPTS:
            template = PromptTemplate.load(prompt)
            thinking = model.startswith(ADAPTIVE_THINKING)
            for batch in (False, True):
                for cache in (False, True):
                    est = estimate_cost(
                        template, texts, model, thinking,
                        batch=batch, prompt_cache=cache,
                        shared_prefix_tokens=prefixes[(tier, prompt)],
                    )
                    rows.append({
                        "tier": tier,
                        "model": model,
                        "prompt": prompt.replace("_3class", ""),
                        "calls": est["n_calls"],
                        "batch": "yes" if batch else "no",
                        "cache": ("yes" if est["cache_used"]
                                  else ("n/a" if cache else "no")),
                        "input_$": round(est["input_usd"], 2),
                        "output_$": round(est["output_usd"], 2),
                        "total_$": round(est["total_usd"], 2),
                    })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("FULL GRID")
    print("=" * 100)
    print(df.to_string(index=False))

    # cheapest viable configuration per tier+prompt
    print("\n" + "=" * 100)
    print("BEST AVAILABLE PER CONDITION, vs the naive baseline")
    print("=" * 100)
    best_rows = []
    for (tier, prompt), grp in df.groupby(["tier", "prompt"], sort=False):
        naive = grp[(grp["batch"] == "no") & (grp["cache"] != "yes")]["total_$"].iloc[0]
        best = grp.loc[grp["total_$"].idxmin()]
        best_rows.append({
            "tier": tier,
            "prompt": prompt,
            "naive_$": naive,
            "best_$": best["total_$"],
            "saving": f"{100 * (1 - best['total_$'] / naive):.0f}%",
            "via": f"batch={best['batch']}, cache={best['cache']}",
        })
    best = pd.DataFrame(best_rows)
    print(best.to_string(index=False))
    print(f"\n  all four conditions, naive: ${best['naive_$'].sum():.2f}")
    print(f"  all four conditions, best:  ${best['best_$'].sum():.2f}")

    step1 = best[(best["tier"] == "corpus") & (best["prompt"] == "zero_shot")]
    print(f"\n  STEP 1 ONLY (corpus / zero-shot / batched): "
          f"${step1['best_$'].iloc[0]:.2f}")

    df.to_csv(config.RESULTS_DIR / "llm_cost_grid.csv", index=False)
    print(f"\nWritten: {config.RESULTS_DIR / 'llm_cost_grid.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
