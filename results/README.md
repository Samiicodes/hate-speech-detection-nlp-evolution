# results

What each file in this directory holds and which script writes it. Everything
here is regenerated from `data/` by the scripts named below, so a file can be
deleted and rebuilt. `results_binary/` is the same shape but holds the archived
outputs of the earlier binary task.

Paths in the "written by" column are `python -m scripts.<name>`.

## Per-system reports

One JSON per system, all written through `src/harness.py` so they share a
schema: headline accuracy and macro-F1, per-class precision, recall and F1, the
confusion matrix, per-target-group blocks with Wilson intervals, HateCheck
results, cost metadata, bootstrap confidence intervals, and the raw test-set
predictions.

| file | written by |
|---|---|
| `rule_based_lexicon.json` | `run_rule_based` |
| `naive_bayes.json` | `run_naive_bayes` |
| `tfidf_linear_svc.json` | `run_classical_ml` |
| `tfidf_logistic_regression.json` | `run_classical_ml --classifier logistic_regression` |
| `roberta_base.json` | `run_transformer`, averaged over three seeds |
| `runs/roberta_base_seed{42,43,44}.json` | `run_transformer`, one per seed |

## Comparison tables

| file | holds | written by |
|---|---|---|
| `comparison.csv` | every system's headline and promoted metrics, one row each | `make_comparison_table` |
| `comparison.md` | the same table as markdown | `make_comparison_table` |

## Out-of-fold predictions

`crossval/<model>_oof.json` holds five-fold out-of-fold predictions over all
19,229 posts, with the post ids, gold labels and per-seed predictions. These
are what the target-group analysis reads, rather than the held-out split.
Written by `src/crossval.py` when called from `run_crossval`,
`run_label_correction` or `run_dose_response`. Files ending `_corrected_oof`
are the reweighted counterparts, and the `dose_*` files are the four
corpus-exposure conditions.

## Target-group analysis

| file | holds | written by |
|---|---|---|
| `group_hate_share.csv` | hate-share per community for the humans and every system, plus RoBERTa's seed sd | `analyse_target_groups` |
| `group_slur_presence.csv` | explicit slur rate per community among hostile posts | `analyse_target_groups` |
| `group_annotator_agreement.csv` | unanimity, pairwise agreement and the hate/offensive split per community | `analyse_target_groups` |
| `group_gradient_agreement.csv` | Spearman rho, p, and mean shift against the human ordering, one row per system | `analyse_target_groups` |
| `group_directional_<model>.csv` | escalation and de-escalation counts and rates per community, one file per system | `analyse_target_groups` |
| `group_escalation.csv` | the long-form escalation table the figures are drawn from | `make_figures` |
| `confound_slur_presence.png` | hate-share against slur rate | `plot_confound` |
| `confound_annotator_agreement.png` | hate-share against annotator agreement | `plot_confound` |

## Label correction

| file | holds | written by |
|---|---|---|
| `label_correction_<model>.json` | baseline and corrected rho, spread and macro-F1, plus the IPF weights summary | `run_label_correction --model <model>` |
| `label_correction_<model>.csv` | per-community baseline, corrected and change, one row per group | `run_label_correction --model <model>` |
| `corrected_features.csv` | token, corrected weight, baseline weight, delta and hand-assigned category | `inspect_corrected_features` |
| `corrected_feature_analysis.json` | the counts and shares behind that categorisation | `inspect_corrected_features` |

Models covered are `naive_bayes`, `tfidf_linear_svc` and `roberta_base`.

## Dose-response

| file | holds | written by |
|---|---|---|
| `dose_response.csv` | one row per corpus-exposure condition with rho, spread and F1 cost | `run_dose_response` |
| `dose_response.json` | the same with the per-condition run metadata | `run_dose_response` |
| `dose_response.png` | the two measures plotted against exposure | `plot_dose_response` |

## HateCheck

| file | holds | written by |
|---|---|---|
| `hatecheck_all_identities.json` | per-functionality pass and flag rates for every identity and system | `run_hatecheck_all` |
| `hatecheck_all_identities.csv` | the flat table of the same | `run_hatecheck_all` |
| `hatecheck_identity_spread_{strict,lenient}.csv` | per-system spread across identities under each label mapping | `report_hatecheck_all --mapping <m>` |
| `identity_intervals_{strict,lenient}.csv` | false-alarm rates on plain identity cases with Wilson intervals | `report_identity_intervals --mapping <m>` |
| `identity_term_composition.csv` | posts and class counts per identity term | `diagnose_identity_terms` |
| `identity_term_weights.csv` | SVC weight and rank, and Naive Bayes log-ratio, for neutral against slur terms | `diagnose_identity_terms` |

`run_llm_hatecheck` adds the LLM's rows to `hatecheck_all_identities.json`.

## Measuring Hate Speech transfer

| file | holds | written by |
|---|---|---|
| `mhs_group_severity.csv` | per-community annotation counts and severity scores in MHS, with the reliability floor flag | `mhs_feasibility` |
| `mhs_irt_check.csv` | the community ordering recomputed from MHS raw ordinals rather than the IRT score | `mhs_irt_check` |
| `mhs_transfer_performance.csv` | macro-F1 for each HateXplain-trained system scored on MHS | `run_mhs_transfer` |
| `mhs_transfer_correlations.csv` | the three Spearman correlations per system | `run_mhs_transfer` |
| `mhs_transfer_hate_share.csv` | per-community hate-share on MHS for each system | `run_mhs_transfer` |
| `mhs_llm_predictions.parquet` | the LLM's per-comment MHS predictions | `run_mhs_transfer` |
| `mhs_logreg_performance.csv`, `mhs_logreg_hate_share.csv`, `mhs_logreg_predictions.parquet` | the same three outputs for logistic regression, which was added later | `mhs_logreg` |
| `mhs_roberta_seed{42,43,44}.parquet` | cached per-seed MHS predictions, so a rerun skips fine-tuning | `mhs_roberta_seeds` |
| `mhs_roberta_seed_stability.csv` | how far the MHS transfer result moves across the three seeds | `mhs_roberta_seeds` |
| `mhs_roberta_seed_hate_share.csv` | per-community hate-share per seed | `mhs_roberta_seeds` |

## Figures

`figures/` holds vector PDF, 300 dpi PNG and a greyscale proof PNG for each
figure, written by `make_figures` and `make_transfer_figures`. Both scripts
take their styling from `src/figstyle.py`, so no figure defines its own
colours. `figures/captions.md` and `figures/captions_transfer.md` hold the
caption text, written by the same two scripts. `figures_final/` at the repo
root holds the subset selected for the dissertation, copied by hand.

## Costs

`llm_cost_grid.csv` holds the priced grid of tier, model, prompt, call count
and input, output and total dollars, with and without batching and caching.
Written by `cost_llm`, which runs without spending anything.

## News probe

A pre-registered out-of-domain check, reported in the dissertation as a
negative result. `run_news_probe` samples 1,000 articles from the
CNN/DailyMail test split at seed 42, splits them into sentences, flags the
sentences mentioning each community, and scores every system over them.

| file | holds | written by |
|---|---|---|
| `preregistration_news_probe.md` | the pre-registration, written by hand before the run |  |
| `news_probe.csv` | per-system flag rate by community with Wilson intervals | `run_news_probe` |
| `news_probe.json` | the same plus run metadata, the sampled article rows and the seed | `run_news_probe` |
| `news_probe_sentences.parquet` | one row per sentence: article id and row, sentence index, the community flags, and each trained system's prediction | `run_news_probe` |
| `news_probe_llm.parquet` | the subset also scored by the LLM, same columns plus `pred_llm` | an earlier revision of `run_news_probe`; no current script regenerates it |

Both parquets ship in the submission archive with the `text` column removed,
to avoid redistributing CNN/DailyMail copy. The text is recoverable from the
public source. `article_id` is that dataset's own SHA-1 story id and
`article_row` indexes `abisee/cnn_dailymail`, config `3.0.0`, split `test`.
`sent_idx` is the position in the output of `sentences()` in
`scripts/run_news_probe.py`, which strips a leading `(CNN)` marker, splits on
`(?<=[.!?])\s+(?=[A-Z"'])` and keeps the pieces over 15 characters. That
function is deterministic and the article sample is drawn with
`numpy.random.default_rng(42)`, so re-running it reproduces the same
sentences in the same order.

## Dropped summarisation study

The perpetrator-framing summarisation study was cut when the project was
reframed. Its volume check ran first and the outputs are kept as data, but the
check script has been removed and nothing regenerates them.

| file | holds |
|---|---|
| `summarisation_volume.json` | the group keyword lists and the article counts behind the volume check |
| `term_group_composition_hatexplain.csv` | posts and class counts per term group in HateXplain |
