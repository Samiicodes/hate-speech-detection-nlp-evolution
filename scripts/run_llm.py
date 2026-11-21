from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from src import config, crossval, data
from src.harness import EvalInputs, evaluate, print_report
from src.models.llm import PRICING, LLMDetector, PromptTemplate, ResponseCache

# Defaults per tier
TIERS = {
    "corpus": {
        "model": "claude-haiku-4-5",
        "scope": "full 19,229-post corpus (out-of-fold not required -- the "
                 "LLM never trains, so every post is already held out)",
    },
    "heldout": {
        "model": "claude-opus-5",
        "scope": "held-out test split only (2,883 posts) + HateCheck",
    },
}

# Rough tokens-per-character for English social media text.
CHARS_PER_TOKEN = 3.6
# Output tokens per classification: one word, plus thinking overhead.
OUTPUT_TOKENS_ZERO_THINKING = 8
OUTPUT_TOKENS_WITH_THINKING = 260

# Batch API is half price
BATCH_DISCOUNT = 0.50

# Minimum cacheable prefix
MIN_CACHEABLE_TOKENS = 1024

# cache write 1.25x, read 0.1x
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def estimate_cost(
    template: PromptTemplate,
    texts: list[str],
    model: str,
    thinking: bool,
    batch: bool = False,
    prompt_cache: bool = False,
    shared_prefix_tokens: int | None = None,
) -> dict:
    in_price, out_price = PRICING.get(model, (0.0, 0.0))
    if batch:
        in_price *= BATCH_DISCOUNT
        out_price *= BATCH_DISCOUNT

    fixed_chars = len(template.system) + sum(
        len(template.user.format(text=e["text"])) + len(e["label"])
        for e in template.examples
    )
    prefix_tokens = (
        shared_prefix_tokens if shared_prefix_tokens is not None
        else fixed_chars / CHARS_PER_TOKEN
    )
    variable_tokens = float(np.sum([
        len(template.user.format(text=t)) / CHARS_PER_TOKEN for t in texts
    ]))
    n = len(texts)

    cache_eligible = prefix_tokens >= MIN_CACHEABLE_TOKENS
    cache_used = prompt_cache and cache_eligible

    if cache_used:
        # one write, then reads; the per-post text is never cacheable
        prefix_usd = (
            prefix_tokens * CACHE_WRITE_MULTIPLIER
            + prefix_tokens * (n - 1) * CACHE_READ_MULTIPLIER
        ) / 1e6 * in_price
    else:
        prefix_usd = prefix_tokens * n / 1e6 * in_price
    variable_usd = variable_tokens / 1e6 * in_price

    out_each = OUTPUT_TOKENS_WITH_THINKING if thinking else OUTPUT_TOKENS_ZERO_THINKING
    total_out = out_each * n

    return {
        "n_calls": n,
        "prefix_tokens": int(prefix_tokens),
        "input_tokens": int(prefix_tokens * n + variable_tokens),
        "output_tokens": int(total_out),
        "input_usd": prefix_usd + variable_usd,
        "output_usd": total_out / 1e6 * out_price,
        "total_usd": prefix_usd + variable_usd + total_out / 1e6 * out_price,
        "model": model,
        "thinking": thinking,
        "batch": batch,
        "cache_eligible": cache_eligible,
        "cache_used": cache_used,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="zero_shot_3class",
                        help="a template name in prompts/")
    parser.add_argument("--tier", default="heldout", choices=list(TIERS))
    parser.add_argument("--provider", default="anthropic",
                        choices=["anthropic", "mock"])
    parser.add_argument("--model", default=None,
                        help="overrides the tier default")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--effort", default="low",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--unparseable-policy", default="normal",
        choices=["hatespeech", "offensive", "normal", "raise"],
        help="prediction used for responses that cannot be read; always counted",
    )
    parser.add_argument("--no-batch", action="store_true",
                        help="use the synchronous API; batch is 50%% cheaper")
    parser.add_argument("--prompt-cache", action="store_true",
                        help="only pays off above the minimum cacheable prefix")
    parser.add_argument("--refusal-policy", default=None,
                        choices=["hatespeech", "offensive", "normal"],
                        help="class a refusal occupies; counted separately "
                             "regardless (default: same as unparseable)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="score a subset (pilot runs)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the cost estimate and stop")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    tier = TIERS[args.tier]
    model = args.model or tier["model"]
    thinking = not args.no_thinking

    task = config.DEFAULT_TASK
    train, val, test = data.get_dataset(task=task)
    hatecheck = data.hatecheck_muslim()
    corpus = data.labelled_hatexplain(task=task).sort_values(
        "post_id", kind="mergesort"
    ).reset_index(drop=True)

    template = PromptTemplate.load(args.prompt)

    # what this tier will classify
    if args.tier == "corpus":
        targets = corpus["text"].tolist()
        label = f"full corpus ({len(corpus)})"
    else:
        targets = (
            val["text"].tolist() + test["text"].tolist()
            + hatecheck["test_case"].tolist()
        )
        label = (f"val {len(val)} + test {len(test)} + hatecheck "
                 f"{len(hatecheck)}")
    if args.limit:
        targets = targets[: args.limit]
        label += f"  [limited to {args.limit}]"

    # Do not price thinking never sent
    from src.models.llm import ADAPTIVE_THINKING

    thinking_effective = thinking and model.startswith(ADAPTIVE_THINKING)
    batch = not args.no_batch
    est = estimate_cost(
        template, targets, model, thinking_effective,
        batch=batch, prompt_cache=args.prompt_cache,
    )
    cache = ResponseCache() if not args.no_cache else None

    print("=" * 78)
    print(f"SYSTEM 4 -- tier '{args.tier}'")
    print("=" * 78)
    print(f"  scope        {tier['scope']}")
    print(f"  classifying  {label}")
    print(f"  prompt       {template.name} v{template.version} "
          f"({len(template.examples)} examples)")
    print(f"  model        {model}")
    if thinking_effective:
        print(f"  thinking     adaptive, effort {args.effort}")
    elif thinking:
        print(f"  thinking     NOT SENT -- {model} does not accept adaptive "
              "thinking; running without it")
    else:
        print("  thinking     disabled by --no-thinking")
    print(f"  temperature  {args.temperature} (requested)")
    print(f"  batch API    {'yes -- 50% off' if batch else 'no (synchronous)'}")
    if args.prompt_cache and not est["cache_eligible"]:
        print(f"  prompt cache requested but the shared prefix is only "
              f"{est['prefix_tokens']} tokens; below the minimum, so not used")
    elif est["cache_used"]:
        print(f"  prompt cache yes -- {est['prefix_tokens']}-token shared prefix")
    print(f"\n  estimated    {est['n_calls']} calls")
    print(f"               {est['input_tokens']:,} input tokens "
          f"-> ${est['input_usd']:.2f}")
    print(f"               {est['output_tokens']:,} output tokens "
          f"-> ${est['output_usd']:.2f}")
    print(f"               TOTAL ~${est['total_usd']:.2f}")
    if cache is not None:
        print(f"  cache holds  {len(cache)} responses "
              "(hits cost nothing and are not re-estimated)")

    if args.dry_run:
        print("\nNo calls made. Drop --dry-run to execute.")
        return 0

    name = args.name or f"llm_{template.name}_{model}_{args.tier}".replace(".", "_")
    provider_kwargs: dict = {"model": model}
    if args.provider == "anthropic":
        provider_kwargs.update(
            temperature=args.temperature,
            effort=args.effort,
            max_tokens=args.max_tokens,
            thinking=thinking,
            prompt_cache=args.prompt_cache and est["cache_eligible"],
        )

    detector = LLMDetector(
        name=name,
        prompt=args.prompt,
        provider=args.provider,
        provider_kwargs=provider_kwargs,
        task=task,
        unparseable_policy=args.unparseable_policy,
        refusal_policy=args.refusal_policy,
        batch=batch and args.provider == "anthropic",
        max_workers=args.max_workers,
        use_cache=not args.no_cache,
    )

    # corpus tier: predictions over everything
    if args.tier == "corpus":
        print(f"\nClassifying {len(targets)} posts...")
        started = time.perf_counter()
        preds = detector.predict(targets)
        elapsed = time.perf_counter() - started

        sub = corpus.iloc[: len(preds)]
        p = detector.params()
        path = crossval.save_oof(
            name, sub, {0: preds}, crossval.load_folds(), task=task,
            notes=(
                "LLM predictions over the full corpus. Not cross-validated: "
                "the model never trains, so every post is already unseen. "
                f"Prompt {template.name} v{template.version}, model {model}."
            ),
            extra={
                # Refusals are their own category
                "refused_indices": p["refused_indices"],
                "unparseable_indices": p["unparseable_indices"],
                "n_refused": p["n_refused"],
                "n_unparseable": p["n_unparseable"],
                "refusal_fallback_class": p["refusal_fallback_class"],
                "unparseable_fallback_class": p["unparseable_fallback_class"],
                "provider_config": {
                    k: v for k, v in p.items()
                    if k.startswith(("temperature", "thinking", "effort"))
                },
            },
        )
        agree = float((np.asarray(preds) == sub["label"].to_numpy()).mean())
        print(f"  {elapsed / 60:.1f} min, agreement with human labels {agree:.4f}")
        _print_response_stats(detector)
        print(f"\nWritten: {path}")
        print("Now run:  python -m scripts.analyse_target_groups")
        return 0

    # held-out tier: the full harness
    report = evaluate(
        detector,
        EvalInputs(train=train, val=val, test=test, hatecheck=hatecheck,
                   task=task),
        notes=(
            f"Prompted LLM, no training. Tier '{args.tier}', model {model}, "
            f"prompt {template.name} v{template.version}. Responses cached on "
            "provider+model+params+prompt+text."
        ),
    )
    print_report(report)
    _print_response_stats(detector)
    print(f"\nEstimated API cost: ${report['cost']['api_cost_usd']}")
    print(f"Results written to {config.results_path(name)}")
    return 0


def _print_response_stats(detector: LLMDetector) -> None:
    p = detector.params()
    print("\nResponse handling")
    print(f"  API calls        {p['n_api_calls']}")
    print(f"  cache hits       {p['n_cache_hits']}")
    print(f"  API errors       {p['n_api_errors']}")
    print(f"  unparseable      {p['n_unparseable']}  "
          f"(policy: {p['unparseable_policy']} -> "
          f"{p['unparseable_fallback_class']})")
    print(f"  temperature      requested {p['temperature_requested']}, "
          f"applied {p['temperature_applied']}")
    if not p["temperature_applied"]:
        print(f"    {p['temperature_note']}")
    for sample in p["unparseable_samples"][:5]:
        print(f"    unparseable: {sample['response']!r}")
    for err in p["api_error_samples"][:3]:
        print(f"    error: {err}")


if __name__ == "__main__":
    sys.exit(main())
