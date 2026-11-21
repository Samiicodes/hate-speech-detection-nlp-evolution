# Reproducing the experiments

This is the full procedure behind the dissertation, from downloading the corpora to drawing the figures. The short version is in the [README](../README.md#reproducing-the-results).

## Setup

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). The experiments ran under Python 3.13.9.

```bash
uv sync
```

That covers systems 1 to 3 and every analysis that reads their results. The fine-tuned transformer needs torch and transformers, which are kept out of the locked dependencies because the right torch wheel depends on your CUDA version:

```bash
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers==5.16.1 accelerate==1.14.0
```

The prompted LLM needs an Anthropic API key. Copy `.env.example` to `.env` and fill it in:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`src/env.py` reads that file at import. `.env` is gitignored and must never be committed, and `tests/test_secrets.py` fails the suite if anything shaped like a key reaches a tracked file. An existing environment variable takes precedence over the file, so a stale local copy cannot shadow a key set elsewhere. To exercise the LLM path without a key, pass `--provider mock`, which is what the LLM tests run against.

Without a GPU, `notebooks/roberta_colab.ipynb` rebuilds the same seeded split and runs the same fine-tuning script on a Colab T4.

The news probe fetches CNN/DailyMail through the `datasets` library, an optional extra: `uv sync --extra news-probe`.

## The data

None of the corpora are committed. `data/` is gitignored, and everything downloads on demand into `data/raw/`.

- **HateXplain** (Mathew et al., 2021) is the training corpus: 20,148 posts with three annotator judgements each. `scripts/prepare_data.py` pulls it from the hate-alert GitHub repository, takes the majority label, drops the 919 posts on which all three annotators disagree, and writes the split. No account or token is needed.
- **HateCheck** (Röttger et al., 2021) is the functional test suite: 3,728 hand-built cases. The same script downloads it from paul-rottger/hatecheck-data. It is used for testing only, and `tests/test_data.py` asserts that no HateCheck string appears in the training split.
- **Measuring Hate Speech** (UC Berkeley D-Lab) is the transfer corpus: 39,565 comments from 7,912 annotators covering 42 target identity subgroups. It comes from the Hugging Face hub as `ucberkeley-dlab/measuring-hate-speech`, licensed CC BY 4.0 and not gated, so `huggingface_hub` fetches it without a token. `scripts/mhs_feasibility.py` handles the download, and the other MHS scripts import their loader from it.

HateDay was tried for the transfer experiment first and turned out to be unusable. It distributes tweet IDs rather than text, so using it means rehydrating every tweet through the Twitter API, and any tweet deleted since collection comes back empty.

## The split

The 70/15/15 split was generated at seed 42 while the task was still binary, and every reported result, three-class included, was scored on it. `prepare_data` therefore always stratifies on the binary label (`config.SPLIT_TASK`), which reproduces the original split exactly. It refuses to overwrite a split that already exists, and `tests/test_data.py` checks that a fresh three-class build would differ, so the reuse cannot be mistaken for an oversight.

## Steps

Run these from the repository root, in this order. Timings are from an RTX 3060 Ti with 8 GB; anything not marked GPU runs on a CPU.

1. `uv run python -m scripts.prepare_data`. Downloads HateXplain and HateCheck, verifies the loader against thirteen known pilot counts, and writes `data/processed/splits.json`. Under a minute.
2. `uv run python -m scripts.run_rule_based`. Builds the lexicon and scores it. Seconds.
3. `uv run python -m scripts.run_naive_bayes`. Seconds.
4. `uv run python -m scripts.run_classical_ml`, then the same command with `--classifier logistic_regression`. Under ten seconds each.
5. `uv run python -m scripts.run_transformer`. GPU. Fine-tunes RoBERTa-base at three seeds, roughly five minutes per seed.
6. `uv run python -m scripts.run_llm --tier corpus --dry-run` to price the run, then `uv run python -m scripts.run_llm --tier corpus --prompt zero_shot_3class` to spend. About $3.75 through the Batch API and 51 minutes of wall clock, most of it queueing. Repeat with `--prompt few_shot_3class` for the second condition, about $6.77. Responses are cached in `data/cache/llm_responses.sqlite`, so an interrupted run resumes without paying twice.
7. `uv run python -m scripts.make_comparison_table`. Assembles `results/comparison.{md,csv}`. Seconds.
8. `uv run python -m scripts.check_human_baseline`, then `uv run python -m scripts.run_crossval --build-folds`. Seconds.
9. `uv run python -m scripts.run_crossval --all-fast`. Out-of-fold predictions for everything except the transformer. About five minutes on the original machine.
10. `uv run python -m scripts.run_crossval --model roberta_base --seeds 42 43 44`. GPU. Fifteen fold-runs, about 66 minutes.
11. `uv run python -m scripts.analyse_target_groups`. The per-community tables and the confound checks. Under a minute.
12. `uv run python -m scripts.run_label_correction --model naive_bayes`, then `--model tfidf_linear_svc`, then `--model roberta_base`. Roughly one minute, three minutes, and 27 minutes; the last needs a GPU.
13. `uv run python -m scripts.inspect_corrected_features`. Seconds.
14. `uv run python -m scripts.run_dose_response`. GPU. Thirty fold-runs, about 41 minutes, no API spend.
15. `uv run python -m scripts.run_mhs_transfer`, then `uv run python -m scripts.mhs_logreg` and `uv run python -m scripts.mhs_roberta_seeds`. The transfer experiments on Measuring Hate Speech. The last fine-tunes three seeds and needs a GPU, and caches its predictions to Parquet so a rerun is cheap.
16. `uv run python -m scripts.make_figures` and `uv run python -m scripts.make_transfer_figures`. Writes vector PDF and 300 dpi PNG into `results/figures/`. Seconds.

`uv run python -m scripts.cost_llm` prints the full pricing grid for every tier and prompt before anything is spent, and `--limit 200` on `run_llm` runs a pilot for a few cents.

## Checking a run

`uv run python -m scripts.check_reproduction` reruns steps 2 to 4 and compares every test-set prediction with the archived results, restoring the archived files afterwards. CI runs it on every push. These reference values come from the archived results:

| Quantity | Value | Written to |
| --- | --- | --- |
| Test split size | 2,883 | `data/processed/splits.json` |
| Macro-F1, rule-based lexicon | 0.4103 | `results/rule_based_lexicon.json` |
| Macro-F1, naive Bayes | 0.6075 | `results/naive_bayes.json` |
| Macro-F1, TF-IDF + linear SVC | 0.6312 | `results/tfidf_linear_svc.json` |
| Macro-F1, TF-IDF + logistic regression | 0.6400 | `results/tfidf_logistic_regression.json` |
| Macro-F1, RoBERTa-base, mean of three seeds | 0.6791 | `results/roberta_base.json` |
| Spearman rho against the human gradient, linear SVC | 0.9879 | `results/group_gradient_agreement.csv` |
| Spearman rho against the human gradient, zero-shot LLM | 0.9030 | `results/group_gradient_agreement.csv` |
| Human hate-share, Jewish-targeted posts | 0.7722 | `results/group_hate_share.csv` |

Systems 1 to 3 reproduce these exactly, including their out-of-fold predictions over all 19,229 posts. RoBERTa does not, for the reason below.

## Known limitations

The experiments ran before the code was under version control, so no commit corresponds to the reported results. The numbers in the dissertation were produced by the code as it stood at the time of each run. The scripts and their outputs are all here, but the exact state that produced any given file cannot be checked out.

RoBERTa is not bit-reproducible at a fixed seed. cuDNN picks convolution and matmul kernels by autotuning, and that choice is not deterministic, so two runs at seed 42 on the same machine can differ in the fourth decimal. That is why the transformer is reported as a mean over three seeds. Passing `--deterministic` to `scripts/run_transformer.py` makes it exact at a cost of roughly 20% throughput.

The prompted LLM was `claude-haiku-4-5`, run between 1 and 3 September 2026. A later run of the same model may not return the same responses.

## The test suite

There are 276 tests; 263 run without PyTorch, because `tests/test_transformer.py` skips itself when torch is missing. Run them with `uv run python -m pytest`. They check the data loader against the thirteen known pilot counts, assert that the test split never reaches a model during training and that no HateCheck string appears in the training split, pin the rule-based lexicon's confusion in both directions so a silent change fails loudly, hold the binary and three-class task designs apart so their results cannot be confused by a script, and fail if anything credential-shaped appears in a tracked file.

Real fine-tuning runs are marked `slow` and deselected by default. Add `-m slow` to include them.
