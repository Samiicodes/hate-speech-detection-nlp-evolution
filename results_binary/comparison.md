# Cross-model comparison

Models: 4 | seed: 42 | task: hate_only (hatespeech = 1; offensive and normal = 0)

## SUMMARY -- slice gaps and the four divergence functionalities

| model                     |   overflag_islam |   overflag_rest |   overflag_gap |   F1_islam |   F1_rest |   F1_gap |   ident_pos_nh |   ident_neutral_nh |   counter_quote_nh |   counter_ref_nh |   n_seeds |
|:--------------------------|-----------------:|----------------:|---------------:|-----------:|----------:|---------:|---------------:|-------------------:|-------------------:|-----------------:|----------:|
| rule_based_lexicon        |           0.5255 |          0.7302 |        -0.2047 |     0.6527 |    0.4418 |   0.2109 |         1.0000 |             1.0000 |             0.4231 |           0.4286 |  nan      |
| tfidf_linear_svc          |           0.2774 |          0.1249 |         0.1524 |     0.7220 |    0.8032 |  -0.0812 |         0.6667 |             0.8889 |             0.5385 |           0.7619 |  nan      |
| tfidf_logistic_regression |           0.2847 |          0.1260 |         0.1587 |     0.7186 |    0.8030 |  -0.0844 |         0.5556 |             0.8889 |             0.4231 |           0.7619 |  nan      |
| roberta_base              |           0.3431 |          0.1434 |         0.1996 |     0.7248 |    0.8206 |  -0.0958 |         1.0000 |             1.0000 |             0.7692 |           0.9206 |    3.0000 |

## Headline (HateXplain test split)

| model                     |   accuracy |   macro_F1 | macro_F1_95CI   |   hate_P |   hate_R |   hate_F1 |   over_flag |   under_detect |
|:--------------------------|-----------:|-----------:|:----------------|---------:|---------:|----------:|------------:|---------------:|
| rule_based_lexicon        |     0.4690 |     0.4658 | [0.447, 0.483]  |   0.3550 |   0.8841 |    0.5066 |      0.7161 |         0.1159 |
| tfidf_linear_svc          |     0.8266 |     0.7992 | [0.785, 0.815]  |   0.7094 |   0.7413 |    0.7250 |      0.1354 |         0.2587 |
| tfidf_logistic_regression |     0.8259 |     0.7986 | [0.784, 0.814]  |   0.7074 |   0.7424 |    0.7245 |      0.1369 |         0.2576 |
| roberta_base              |     0.8354 |     0.8152 | [0.800, 0.830]  |   0.6994 |   0.8185 |    0.7541 |      0.1571 |         0.1815 |

## Islam-targeted slice vs rest

| model                     |   islam_n |   islam_macro_F1 |   islam_hate_R |   islam_over_flag |   islam_under_detect |   rest_macro_F1 |   gap_macro_F1 |
|:--------------------------|----------:|-----------------:|---------------:|------------------:|---------------------:|----------------:|---------------:|
| rule_based_lexicon        |  316.0000 |           0.6527 |         0.8324 |            0.5255 |               0.1676 |          0.4418 |         0.2109 |
| tfidf_linear_svc          |  316.0000 |           0.7220 |         0.7263 |            0.2774 |               0.2737 |          0.8032 |        -0.0812 |
| tfidf_logistic_regression |  316.0000 |           0.7186 |         0.7263 |            0.2847 |               0.2737 |          0.8030 |        -0.0844 |
| roberta_base              |  316.0000 |           0.7248 |         0.7896 |            0.3431 |               0.2104 |          0.8206 |        -0.0958 |

## HateCheck, Muslim-targeted

| model                     |   n_cases |   recall_373_hateful |   false_alarm_111_nonhateful |   accuracy |
|:--------------------------|----------:|---------------------:|-----------------------------:|-----------:|
| rule_based_lexicon        |  484.0000 |               0.2279 |                       0.3153 |     0.3326 |
| tfidf_linear_svc          |  484.0000 |               0.4906 |                       0.3243 |     0.5331 |
| tfidf_logistic_regression |  484.0000 |               0.5201 |                       0.3874 |     0.5413 |
| roberta_base              |  484.0000 |               0.5121 |                       0.1081 |     0.5992 |

## HateCheck per-functionality pass rate

| functionality      | gold        |   rule_based_lexicon |   tfidf_linear_svc |   tfidf_logistic_regression |   roberta_base |
|:-------------------|:------------|---------------------:|-------------------:|----------------------------:|---------------:|
| spell_char_swap_h  | hateful     |               0.0000 |             0.5263 |                      0.5263 |         0.2105 |
| spell_leet_h       | hateful     |               0.0000 |             0.4231 |                      0.4615 |         0.4103 |
| spell_space_del_h  | hateful     |               0.0476 |             0.7143 |                      0.7143 |         0.7302 |
| negate_pos_h       | hateful     |               0.0500 |             0.4000 |                      0.4500 |         0.2833 |
| derog_dehum_h      | hateful     |               0.1000 |             0.7000 |                      0.7000 |         0.6833 |
| derog_neg_emote_h  | hateful     |               0.1000 |             0.3500 |                      0.4000 |         0.5833 |
| spell_char_del_h   | hateful     |               0.1500 |             0.4000 |                      0.4000 |         0.6333 |
| spell_space_add_h  | hateful     |               0.1538 |             0.3462 |                      0.3462 |         0.0385 |
| derog_impl_h       | hateful     |               0.2000 |             0.4000 |                      0.4000 |         0.3333 |
| derog_neg_attrib_h | hateful     |               0.2500 |             0.3500 |                      0.4500 |         0.4167 |
| threat_norm_h      | hateful     |               0.2500 |             0.5000 |                      0.5500 |         0.8167 |
| threat_dir_h       | hateful     |               0.2632 |             0.4737 |                      0.4737 |         0.8947 |
| ref_subs_clause_h  | hateful     |               0.3500 |             0.6000 |                      0.6000 |         0.5167 |
| phrase_question_h  | hateful     |               0.4000 |             0.2500 |                      0.3500 |         0.4500 |
| phrase_opinion_h   | hateful     |               0.4211 |             0.5789 |                      0.6316 |         0.6491 |
| ref_subs_sent_h    | hateful     |               0.4211 |             0.4211 |                      0.4737 |         0.4561 |
| profanity_h        | hateful     |               0.4500 |             0.7500 |                      0.7500 |         0.5333 |
| slur_h             | hateful     |               0.5417 |             0.6667 |                      0.7083 |         0.7083 |
| counter_quote_nh   | non-hateful |               0.4231 |             0.5385 |                      0.4231 |         0.7692 |
| counter_ref_nh     | non-hateful |               0.4286 |             0.7619 |                      0.7619 |         0.9206 |
| negate_neg_nh      | non-hateful |               0.5789 |             0.5789 |                      0.5263 |         0.7719 |
| ident_neutral_nh   | non-hateful |               1.0000 |             0.8889 |                      0.8889 |         1.0000 |
| ident_pos_nh       | non-hateful |               1.0000 |             0.6667 |                      0.5556 |         1.0000 |

## Cost

| model                     |   train_s |   tune_s |   infer_s_per_1000 |   size_MB |   api_cost_usd |
|:--------------------------|----------:|---------:|-------------------:|----------:|---------------:|
| rule_based_lexicon        |    0.0943 |   0.0000 |             0.0086 |    0.0060 |         0.0000 |
| tfidf_linear_svc          |    2.5631 |  22.0653 |             0.1179 |    2.5770 |         0.0000 |
| tfidf_logistic_regression |    2.8976 |  21.7773 |             0.1387 |    2.5770 |         0.0000 |
| roberta_base              |  293.7888 |   0.0000 |             1.2885 |  498.6590 |         0.0000 |

## Run-to-run variability (multi-seed models, mean ± std)

| model        |   n_seeds | seeds      | macro_F1        | overflag_gap     | F1_gap           | ident_pos_nh    | ident_neutral_nh   | counter_quote_nh   | counter_ref_nh   |
|:-------------|----------:|:-----------|:----------------|:-----------------|:-----------------|:----------------|:-------------------|:-------------------|:-----------------|
| roberta_base |         3 | 42, 43, 44 | 0.8152 ± 0.0044 | +0.1996 ± 0.0069 | -0.0958 ± 0.0143 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000    | 0.7692 ± 0.0666    | 0.9206 ± 0.0991  |
