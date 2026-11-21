# Figure captions

Paste into the .tex; the figures carry no titles of their own.

## gradient

```latex
\caption{Share of hostile posts graded as hate speech, by target community and rater. Communities are ordered by the human rate, which spans 48.6 points. All four learned systems reproduce that ordering (Spearman's $\rho$ from $+0.903$ to $+0.988$) while grading more severely than the annotators did in 38 of 40 system--community cells; the two exceptions are RoBERTa on refugee (28.0\% against 32.8\%) and on women (28.9\% against 29.9\%). The rule-based lexicon has no representation of severity and stays between 46.4\% and 55.6\% for nine of the ten communities, the exception being Arab at 71.2\%. Out-of-fold predictions over all 19{,}229 posts; RoBERTa is the mean of three seeds.}
```

## confound

```latex
\caption{Human severity grading against explicit slur content, by target community. No trend line is fitted: Spearman's $\rho = +0.139$ ($p = 0.701$). African, homosexual and women carry slur rates within 3.1 points of one another while their hate-shares span 45.8 points, which rules out content severity as the explanation for the gradient.}
```

## cost

```latex
\caption{Held-out macro-F1 against training wall-clock (log scale). RoBERTa buys 4.8 macro-F1 points over the TF-IDF SVM for 73$\times$ the training time. The prompted LLM does not train at all and so has no position on the time axis; it is shown as a horizontal reference at its own macro-F1. All five scores are on the same held-out split of 2,883 posts.}
```

## escalation

```latex
\caption{Escalation rate --- the share of posts human annotators called \emph{offensive} that the system called \emph{hate speech} --- by target community, with Wilson 95\% intervals. Communities are ordered by RoBERTa's rate, which runs from 56.6\% for Hispanic targets to 7.9\% for women, a factor of 7.2 with non-overlapping intervals. The SVM tracks that ordering closely ($\rho = +0.952$); the rule-based lexicon does not track it at all ($\rho = +0.055$) and spans only a factor of 1.9, escalating on 37.2\% of human-offensive posts about women where RoBERTa escalates on 7.9\%.}
```

## label_dependence

```latex
\caption{Left: the share of each system's group ordering removed by correcting the corpus labels, across four architectures. Right: the same intervention applied within a single architecture (\texttt{roberta-base} throughout) at four levels of corpus exposure. Both series in the right-hand panel are percentages on one shared axis; note that the two panels use different vertical ranges. The two measures of label-dependence move in opposite directions on the right, so the corpus-exposure account is not supported by the within-architecture test.}
```
