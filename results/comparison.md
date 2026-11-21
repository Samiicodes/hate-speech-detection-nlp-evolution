# Cross-model comparison

Models: 5 | seed: 42 | task: three_class (hatespeech = 0; offensive = 1; normal = 2)

## HEADLINE -- hate-share by target group (human vs each model)

Held-out test split only, so most communities fall below the n>=200 reliability bar.
  The properly powered version -- all ten communities on out-of-fold predictions over
  the full corpus, plus the label-free LLM row -- is scripts/analyse_target_groups.py.

| group      |   n_hostile |   human |   rule_based_lexicon |   naive_bayes |   tfidf_linear_svc |   tfidf_logistic_regression |   roberta_base |
|:-----------|------------:|--------:|---------------------:|--------------:|-------------------:|----------------------------:|---------------:|
| hispanic   |          56 |  0.7857 |               0.5116 |        0.7500 |             0.7872 |                      0.8372 |         0.9058 |
| jewish     |         264 |  0.7765 |               0.5776 |        0.9130 |             0.8643 |                      0.8789 |         0.8337 |
| african    |         446 |  0.7534 |               0.5610 |        0.7995 |             0.8295 |                      0.8130 |         0.8171 |
| arab       |          93 |  0.7204 |               0.7333 |        0.8068 |             0.8046 |                      0.7753 |         0.7484 |
| islam      |         271 |  0.6605 |               0.5656 |        0.8485 |             0.7854 |                      0.7617 |         0.7369 |
| asian      |          45 |  0.4667 |               0.5185 |        0.5484 |             0.5806 |                      0.6207 |         0.5474 |
| homosexual |         280 |  0.4250 |               0.5641 |        0.5484 |             0.5314 |                      0.4777 |         0.4917 |
| refugee    |          81 |  0.3827 |               0.5600 |        0.2955 |             0.2553 |                      0.2453 |         0.3095 |
| women      |         203 |  0.3005 |               0.6082 |        0.3497 |             0.2857 |                      0.2439 |         0.2720 |
| caucasian  |          57 |  0.2807 |               0.5636 |        0.4643 |             0.3600 |                      0.3478 |         0.3146 |

## Rank agreement with the human severity gradient

Read rho next to macro-F1 below: a model that cannot grade severity produces a flat
  profile, and a flat profile is uncorrelated with any gradient. See 'Flat is not fair'.

| model                     |   n_groups |   spearman_rho |   p_value |   mean_shift |
|:--------------------------|-----------:|---------------:|----------:|-------------:|
| rule_based_lexicon        |     5.0000 |        -0.3000 |    0.6238 |      -0.0079 |
| naive_bayes               |     5.0000 |         0.9000 |    0.0833 |       0.1086 |
| tfidf_linear_svc          |     5.0000 |         1.0000 |    0.0000 |       0.0761 |
| tfidf_logistic_regression |     5.0000 |         1.0000 |    0.0167 |       0.0518 |
| roberta_base              |     5.0000 |         0.9667 |    0.0389 |       0.0471 |

## Per-slice gaps and the four divergence functionalities

| model                     |   islam_human_share |   islam_model_share |   islam_share_shift |   islam_escalation |   islam_de_escalation |   F1_islam |   F1_rest |   F1_gap |   ident_pos_nh |   ident_neutral_nh |   counter_quote_nh |   counter_ref_nh |   n_seeds |
|:--------------------------|--------------------:|--------------------:|--------------------:|-------------------:|----------------------:|-----------:|----------:|---------:|---------------:|-------------------:|-------------------:|-----------------:|----------:|
| rule_based_lexicon        |              0.6605 |              0.5656 |             -0.0949 |             0.2283 |                0.3128 |     0.4163 |    0.4055 |   0.0108 |         1.0000 |             1.0000 |             0.7692 |           0.9524 |  nan      |
| naive_bayes               |              0.6605 |              0.8485 |              0.1880 |             0.4130 |                0.0782 |     0.4409 |    0.6202 |  -0.1793 |         0.6296 |             0.8333 |             0.5000 |           0.6667 |  nan      |
| tfidf_linear_svc          |              0.6605 |              0.7854 |              0.1248 |             0.3478 |                0.1006 |     0.4527 |    0.6455 |  -0.1928 |         0.6667 |             0.8889 |             0.4615 |           0.7619 |  nan      |
| tfidf_logistic_regression |              0.6605 |              0.7617 |              0.1012 |             0.3152 |                0.1006 |     0.4872 |    0.6513 |  -0.1641 |         0.5556 |             0.8333 |             0.3462 |           0.7143 |  nan      |
| roberta_base              |              0.6605 |              0.7369 |              0.0764 |             0.4493 |                0.1304 |     0.5493 |    0.6889 |  -0.1396 |         1.0000 |             1.0000 |             0.7692 |           0.9048 |    3.0000 |

## Overall performance (HateXplain test split)

| model                     |   accuracy |   macro_F1 | macro_F1_95CI   |   hatespeech_P |   hatespeech_R |   hatespeech_F1 |   offensive_P |   offensive_R |   offensive_F1 |   normal_P |   normal_R |   normal_F1 |
|:--------------------------|-----------:|-----------:|:----------------|---------------:|---------------:|----------------:|--------------:|--------------:|---------------:|-----------:|-----------:|------------:|
| rule_based_lexicon        |     0.4103 |     0.4103 | [0.393, 0.427]  |         0.4228 |         0.5298 |          0.4703 |        0.3018 |        0.3843 |         0.3381 |     0.5680 |     0.3363 |      0.4225 |
| naive_bayes               |     0.6230 |     0.6075 | [0.590, 0.625]  |         0.6670 |         0.7255 |          0.6950 |        0.5277 |        0.3970 |         0.4531 |     0.6382 |     0.7150 |      0.6745 |
| tfidf_linear_svc          |     0.6459 |     0.6312 | [0.615, 0.648]  |         0.7141 |         0.7278 |          0.7209 |        0.5534 |        0.4201 |         0.4776 |     0.6450 |     0.7540 |      0.6952 |
| tfidf_logistic_regression |     0.6476 |     0.6400 | [0.624, 0.657]  |         0.7183 |         0.7255 |          0.7219 |        0.5483 |        0.4861 |         0.5153 |     0.6579 |     0.7097 |      0.6828 |
| roberta_base              |     0.6835 |     0.6791 | [0.663, 0.696]  |         0.7209 |         0.7990 |          0.7579 |        0.5612 |        0.5687 |         0.5648 |     0.7526 |     0.6805 |      0.7146 |

## Worked slice: Islam-targeted vs rest (one community of the ten)

| model                     |   islam_n |   islam_macro_F1 |   rest_macro_F1 |   gap_macro_F1 |   islam_hatespeech_R |   islam_offensive_R |   islam_normal_R |
|:--------------------------|----------:|-----------------:|----------------:|---------------:|---------------------:|--------------------:|-----------------:|
| rule_based_lexicon        |  316.0000 |           0.4163 |          0.4055 |         0.0108 |               0.5196 |              0.3152 |           0.5111 |
| naive_bayes               |  316.0000 |           0.4409 |          0.6202 |        -0.1793 |               0.6872 |              0.1304 |           0.7556 |
| tfidf_linear_svc          |  316.0000 |           0.4527 |          0.6455 |        -0.1928 |               0.6872 |              0.1848 |           0.6667 |
| tfidf_logistic_regression |  316.0000 |           0.4872 |          0.6513 |        -0.1641 |               0.6983 |              0.2717 |           0.6222 |
| roberta_base              |  316.0000 |           0.5493 |          0.6889 |        -0.1396 |               0.7672 |              0.3188 |           0.6519 |

## HateCheck (Muslim-targeted subset, the only one at usable size)

| model                     |   n_cases |   recall_373_hateful |   false_alarm_111_nonhateful |   accuracy |
|:--------------------------|----------:|---------------------:|-----------------------------:|-----------:|
| rule_based_lexicon        |  484.0000 |               0.0241 |                       0.0631 |     0.2335 |
| naive_bayes               |  484.0000 |               0.4718 |                       0.3423 |     0.5145 |
| tfidf_linear_svc          |  484.0000 |               0.4799 |                       0.3604 |     0.5165 |
| tfidf_logistic_regression |  484.0000 |               0.5174 |                       0.4414 |     0.5269 |
| roberta_base              |  484.0000 |               0.3539 |                       0.1171 |     0.4752 |

## HateCheck per-functionality pass rate

| functionality      | gold        |   rule_based_lexicon |   naive_bayes |   tfidf_linear_svc |   tfidf_logistic_regression |   roberta_base |
|:-------------------|:------------|---------------------:|--------------:|-------------------:|----------------------------:|---------------:|
| derog_dehum_h      | hateful     |               0.0000 |        0.8000 |             0.7500 |                      0.7000 |         0.4500 |
| derog_impl_h       | hateful     |               0.0000 |        0.5500 |             0.4500 |                      0.6000 |         0.2500 |
| derog_neg_attrib_h | hateful     |               0.0000 |        0.3500 |             0.3500 |                      0.3500 |         0.1667 |
| derog_neg_emote_h  | hateful     |               0.0000 |        0.6000 |             0.3500 |                      0.4500 |         0.2000 |
| negate_pos_h       | hateful     |               0.0000 |        0.4000 |             0.4500 |                      0.5000 |         0.1167 |
| slur_h             | hateful     |               0.0000 |        0.7500 |             0.7083 |                      0.6667 |         0.6667 |
| spell_char_swap_h  | hateful     |               0.0000 |        0.3158 |             0.3684 |                      0.3684 |         0.1053 |
| spell_leet_h       | hateful     |               0.0000 |        0.3077 |             0.5000 |                      0.4615 |         0.3077 |
| spell_space_add_h  | hateful     |               0.0000 |        0.3846 |             0.3077 |                      0.3462 |         0.0256 |
| spell_space_del_h  | hateful     |               0.0000 |        0.4286 |             0.7143 |                      0.8095 |         0.4286 |
| phrase_question_h  | hateful     |               0.0500 |        0.2500 |             0.2500 |                      0.3000 |         0.3500 |
| profanity_h        | hateful     |               0.0500 |        0.6500 |             0.6000 |                      0.7000 |         0.3833 |
| ref_subs_clause_h  | hateful     |               0.0500 |        0.4500 |             0.5000 |                      0.5000 |         0.3000 |
| spell_char_del_h   | hateful     |               0.0500 |        0.4500 |             0.4500 |                      0.4500 |         0.3333 |
| phrase_opinion_h   | hateful     |               0.0526 |        0.5789 |             0.6842 |                      0.6316 |         0.5965 |
| ref_subs_sent_h    | hateful     |               0.0526 |        0.2632 |             0.2632 |                      0.2632 |         0.2456 |
| threat_dir_h       | hateful     |               0.0526 |        0.4211 |             0.4211 |                      0.4737 |         0.8070 |
| threat_norm_h      | hateful     |               0.1000 |        0.5500 |             0.5000 |                      0.7500 |         0.7000 |
| counter_quote_nh   | non-hateful |               0.7692 |        0.5000 |             0.4615 |                      0.3462 |         0.7692 |
| counter_ref_nh     | non-hateful |               0.9524 |        0.6667 |             0.7619 |                      0.7143 |         0.9048 |
| ident_neutral_nh   | non-hateful |               1.0000 |        0.8333 |             0.8889 |                      0.8333 |         1.0000 |
| ident_pos_nh       | non-hateful |               1.0000 |        0.6296 |             0.6667 |                      0.5556 |         1.0000 |
| negate_neg_nh      | non-hateful |               1.0000 |        0.7368 |             0.4737 |                      0.4211 |         0.7368 |

## Cost

| model                     |   train_s |   tune_s |   infer_s_per_1000 |   size_MB |   api_cost_usd |
|:--------------------------|----------:|---------:|-------------------:|----------:|---------------:|
| rule_based_lexicon        |    0.0962 |   0.0413 |             0.0045 |    0.0060 |         0.0000 |
| naive_bayes               |    0.1763 |   1.5123 |             0.0099 |    0.6390 |         0.0000 |
| tfidf_linear_svc          |    3.8493 |  31.1203 |             0.1361 |    4.0300 |         0.0000 |
| tfidf_logistic_regression |    4.2897 |  34.3849 |             0.1308 |    4.0300 |         0.0000 |
| roberta_base              |  281.7923 |   0.0000 |             1.2432 |  498.6620 |         0.0000 |

## Run-to-run variability (multi-seed models, mean ± std)

| model        |   n_seeds | seeds      | macro_F1        | F1_gap           | islam_model_share   | ident_pos_nh    | ident_neutral_nh   | counter_quote_nh   | counter_ref_nh   |
|:-------------|----------:|:-----------|:----------------|:-----------------|:--------------------|:----------------|:-------------------|:-------------------|:-----------------|
| roberta_base |         3 | 42, 43, 44 | 0.6791 ± 0.0046 | -0.1396 ± 0.0090 | 0.7369 ± 0.0032     | 1.0000 ± 0.0000 | 1.0000 ± 0.0000    | 0.7692 ± 0.0385    | 0.9048 ± 0.0952  |

## Paired comparison: bootstrap CI on macro-F1 difference + McNemar

| model_a                   | model_b                   |   acc_a |   acc_b |   macroF1_diff | diff_95CI          |   CI_width | CI_excludes_0   |   mcnemar_p | sig_at_0.05   |
|:--------------------------|:--------------------------|--------:|--------:|---------------:|:-------------------|-----------:|:----------------|------------:|:--------------|
| rule_based_lexicon        | naive_bayes               |  0.4103 |  0.6230 |        -0.1973 | [-0.2217, -0.1730] |     0.0488 | True            |      0.0000 | True          |
| rule_based_lexicon        | tfidf_linear_svc          |  0.4103 |  0.6459 |        -0.2210 | [-0.2450, -0.1977] |     0.0473 | True            |      0.0000 | True          |
| rule_based_lexicon        | tfidf_logistic_regression |  0.4103 |  0.6476 |        -0.2297 | [-0.2530, -0.2066] |     0.0464 | True            |      0.0000 | True          |
| rule_based_lexicon        | roberta_base              |  0.4103 |  0.6819 |        -0.2683 | [-0.2914, -0.2463] |     0.0451 | True            |      0.0000 | True          |
| naive_bayes               | tfidf_linear_svc          |  0.6230 |  0.6459 |        -0.0237 | [-0.0376, -0.0105] |     0.0271 | True            |      0.0007 | True          |
| naive_bayes               | tfidf_logistic_regression |  0.6230 |  0.6476 |        -0.0325 | [-0.0476, -0.0178] |     0.0298 | True            |      0.0007 | True          |
| naive_bayes               | roberta_base              |  0.6230 |  0.6819 |        -0.0710 | [-0.0900, -0.0525] |     0.0375 | True            |      0.0000 | True          |
| tfidf_linear_svc          | tfidf_logistic_regression |  0.6459 |  0.6476 |        -0.0088 | [-0.0197, +0.0030] |     0.0227 | False           |      0.7941 | False         |
| tfidf_linear_svc          | roberta_base              |  0.6459 |  0.6819 |        -0.0473 | [-0.0651, -0.0306] |     0.0345 | True            |      0.0000 | True          |
| tfidf_logistic_regression | roberta_base              |  0.6476 |  0.6819 |        -0.0385 | [-0.0553, -0.0214] |     0.0339 | True            |      0.0001 | True          |
