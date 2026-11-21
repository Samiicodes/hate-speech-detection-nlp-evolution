"""One module per detector paradigm, in the order the brief lists them.

  rule_based     -- system 1, rule-based: corpus-derived lexicon
  naive_bayes    -- system 2, statistical: multinomial NB over term counts
  classical_ml   -- system 3, machine learning: TF-IDF + linear classifier
  transformer    -- system 4, deep learning: fine-tuned RoBERTa
  llm            -- system 5, large language model: prompted, no training
  (guard_model, system 6, to follow)
"""
