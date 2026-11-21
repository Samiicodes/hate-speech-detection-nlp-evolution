# The Evolution of NLP Methods in Hate Speech Detection

## What this is

This repository holds the code and outputs behind my MSc dissertation, "The
Evolution of NLP Methods in Hate Speech Detection: Comparing Rule-Based,
Machine Learning, Deep Learning and LLM Approaches Across Target Groups" (MSc
Artificial Intelligence, Lancaster University). Five detection systems, from a
rule-based lexicon through to a prompted LLM, are trained and scored on the
same posts through the same evaluation harness, then compared across the target
communities their inputs mention. The findings and their discussion are in the
dissertation itself. What follows is setup, data, and how to reproduce a run.

## Requirements and setup

Python 3.11 or newer. I developed on 3.13.9, and the pins in `requirements.txt`
come from that environment.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt              # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # Linux, macOS
```

That covers systems 1 to 3. The fine-tuned Transformer needs torch and
transformers, which I kept out of `requirements.txt` because the right torch
wheel depends on your CUDA version:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.16.1 accelerate==1.14.0
```

The LLM system needs an Anthropic API key. Copy `.env.example` to `.env` and
fill it in:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`src/env.py` reads that file at import. `.env` is gitignored and must never be
committed. An existing environment variable takes precedence over the file, so
a stale local copy can't shadow a key set elsewhere. If you'd rather not use a
key at all, `--provider mock` exercises the whole LLM path, which is what the
LLM tests run against.

With no GPU to hand, `notebooks/roberta_colab.ipynb` rebuilds the same seeded
split and runs the same script on a Colab T4.

## Getting the data

None of the corpora are committed. `data/` is gitignored and everything
downloads on demand into `data/raw/`.

HateXplain (Mathew et al. 2021) is the training corpus, 20,148 posts with three
annotator judgements each. `scripts/prepare_data.py` pulls it from the
hate-alert GitHub repository, takes the majority label, drops the 919 posts
where all three annotators disagree, and writes the splits. No account or token
is needed.

HateCheck (Röttger et al. 2021) is the functional test suite, 3,728 hand-built
cases. The same script downloads it from paul-rottger/hatecheck-data. It is
used for testing only, and a check in `tests/test_data.py` asserts that no
HateCheck string appears in the training split.

Measuring Hate Speech (UC Berkeley D-Lab) is the transfer corpus, 39,565
comments from 7,912 annotators covering 42 target identity subgroups. It comes
from the Hugging Face hub as `ucberkeley-dlab/measuring-hate-speech`, licensed
CC-BY-4.0 and not gated, so `huggingface_hub` will fetch it without a token.
`scripts/mhs_feasibility.py` handles the download and the other MHS scripts
import their loader from it.

I tried HateDay for the transfer experiment first, and it turned out to be
unusable here. It distributes tweet IDs rather than text, so using it means
rehydrating every tweet through the Twitter API, and any tweet deleted since
collection comes back empty. Measuring Hate Speech replaced it.

## How to reproduce

Run these from the repository root, in this order. Timings are from an RTX 3060
Ti with 8 GB; anything not marked GPU runs fine on CPU.

1. `python -m scripts.prepare_data`. Downloads HateXplain and HateCheck,
   verifies the loader against thirteen known pilot counts, and writes
   `data/processed/splits.json`. Under a minute. It refuses to rebuild a split
   that already exists, because the split was stratified on the binary label
   and rebuilding it under the three-class task would reshuffle the rows and
   break comparability with `results_binary/`.
2. `python -m scripts.run_rule_based`. Builds the lexicon and scores it.
   Seconds.
3. `python -m scripts.run_naive_bayes`. Seconds.
4. `python -m scripts.run_classical_ml`, then the same command with
   `--classifier logistic_regression`. Under ten seconds each.
5. `python -m scripts.run_transformer`. GPU. Fine-tunes RoBERTa-base at three
   seeds, roughly five minutes per seed.
6. `python -m scripts.run_llm --tier corpus --dry-run` to price the run, then
   `python -m scripts.run_llm --tier corpus --prompt zero_shot_3class` to
   spend. About $3.75 through the Batch API and 51 minutes of wall clock, most
   of it queueing. Repeat with `--prompt few_shot_3class` for the second
   condition, about $6.77. Responses are cached in
   `data/cache/llm_responses.sqlite`, so an interrupted run resumes without
   paying twice.
7. `python -m scripts.make_comparison_table`. Assembles
   `results/comparison.{md,csv}`. Seconds.
8. `python -m scripts.check_human_baseline`, then
   `python -m scripts.run_crossval --build-folds`. Seconds.
9. `python -m scripts.run_crossval --all-fast`. Out-of-fold predictions for
   everything except the Transformer. About five minutes.
10. `python -m scripts.run_crossval --model roberta_base --seeds 42 43 44`.
    GPU. Fifteen fold-runs, about 66 minutes.
11. `python -m scripts.analyse_target_groups`. The per-community tables and
    the confound checks. Under a minute.
12. `python -m scripts.run_label_correction --model naive_bayes`, then
    `--model tfidf_linear_svc`, then `--model roberta_base`. Roughly one
    minute, three minutes, and 27 minutes; the last needs a GPU.
13. `python -m scripts.inspect_corrected_features`. Seconds.
14. `python -m scripts.run_dose_response`. GPU. Thirty fold-runs, about 41
    minutes, no API spend.
15. `python -m scripts.run_mhs_transfer`, then `python -m scripts.mhs_logreg`
    and `python -m scripts.mhs_roberta_seeds`. The transfer experiments on
    Measuring Hate Speech. The last one fine-tunes three seeds and needs a
    GPU, and caches its predictions to parquet so a rerun is cheap.
16. `python -m scripts.make_figures` and
    `python -m scripts.make_transfer_figures`. Writes vector PDF and 300 dpi
    PNG into `results/figures/`. Seconds.

`python -m scripts.cost_llm` prints the full pricing grid for every tier and
prompt before anything is spent, and `--limit 200` on `run_llm` runs a pilot
for a few cents.

## Repository map

```
data/            downloaded corpora, saved splits and folds (gitignored)
figures_final/   the figures selected for the dissertation, PDF and PNG
notebooks/       Colab notebook for fine-tuning without a local GPU
prompts/         versioned YAML prompt templates for the LLM system
results/         three-class outputs: one JSON per model, plus analyses
results_binary/  archived outputs from the earlier binary task
scripts/         entry points, one per experiment
src/             the harness, data loading, group measures, model wrappers
tests/           pytest suite
```

## Reproduction check

These are for verifying that a run came out the way mine did, not a summary of
what the project found. If your numbers match these, the pipeline is behaving.

| quantity | value | written to |
|---|---|---|
| test split size | 2,883 | `data/processed/splits.json` |
| macro-F1, rule-based lexicon | 0.4103 | `results/rule_based_lexicon.json` |
| macro-F1, Naive Bayes | 0.6075 | `results/naive_bayes.json` |
| macro-F1, TF-IDF + LinearSVC | 0.6312 | `results/tfidf_linear_svc.json` |
| macro-F1, RoBERTa-base, mean of three seeds | 0.6791 | `results/roberta_base.json` |
| Spearman rho vs human gradient, LinearSVC | 0.9879 | `results/group_gradient_agreement.csv` |
| Spearman rho vs human gradient, zero-shot LLM | 0.9030 | `results/group_gradient_agreement.csv` |
| human hate-share, jewish | 0.7722 | `results/group_hate_share.csv` |

Systems 1 to 3 land on these exactly. RoBERTa will not, for the reason below.

## Known limitations

The experiments ran before I put the repository under version control, so
there's no commit that corresponds to the reported results. The numbers in the
dissertation were produced by the code as it stood at the time of each run, and
while the scripts and their outputs are all here, you can't check out the exact
state that produced any given file. Anything from here on is tracked.

RoBERTa is not bit-reproducible at a fixed seed. cuDNN picks convolution and
matmul kernels by autotuning, and that choice isn't deterministic, so two runs
at seed 42 on the same machine can differ in the fourth decimal. That is why
the Transformer is reported as a mean over three seeds rather than a single
number. Passing `--deterministic` to `scripts/run_transformer.py` makes it
exact at a cost of roughly 20% throughput.

## The test suite

There are 276 tests. Run them with `pytest`. They check the data loader against
the thirteen known pilot counts, assert the test split never reaches a model
during training and that no HateCheck string appears in the training split, pin
the rule-based lexicon's confusion in both directions so a silent change fails
loudly, hold the two task designs apart so binary and three-class results can't
be confused by a script, and fail if anything credential-shaped appears in a
tracked file.

Real fine-tuning runs are marked `slow` and deselected by default. Add
`-m slow` to include them.
