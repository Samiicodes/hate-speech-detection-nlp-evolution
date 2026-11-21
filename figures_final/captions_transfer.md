# Captions for the transfer figures

## inversion

```latex
\caption{Macro-F1 on the HateXplain held-out split against macro-F1 on Measuring Hate Speech, for the same six systems run as inference only. A single shared vertical scale, so a crossing means the two systems genuinely change places. The rule-based lexicon is the only system that improves (0.410 to 0.499); both linear variants fall past it (SVM 0.631 to 0.480, logistic regression 0.640 to 0.485), as does Naive Bayes (0.608 to 0.487). The prompted LLM, which never saw HateXplain's labels, moves least (0.605 to 0.581) and finishes first. MHS scores use the `bands` gold mapping; the ordering is the same under the other two mappings tested.}
```

## replication

```latex
\caption{Human severity grading by target community on two corpora: HateXplain's three annotators per post against Measuring Hate Speech's 7,912 annotators, on the same ten communities under the target mapping given in the text. Shared vertical scale. The ordering does not replicate --- Spearman's $\rho = -0.19$ --- and the spread contracts from 48.6 points to 18.7. Jewish falls from first to ninth and women rise from ninth to third; both are drawn in full ink.}
```

## carried

```latex
\caption{What each system's community ordering on MHS text correlates with: the HateXplain annotators, the MHS annotators, or that system's own ordering on HateXplain. The middle series is lowest for every system and negative for three, so models reading MHS text order communities more like the annotators of the corpus they were trained on than like the annotators who labelled the text in front of them. The rule-based lexicon is shaded as a control: it has no learned ordering, and correspondingly carries nothing. RoBERTa's bars are $\pm$1 sd over three seeds; the other systems are deterministic.}
```

## contestedness

```latex
\caption{Human severity grading against annotator contestedness --- the share of a community's hostile posts on which at least one annotator said \emph{hate speech} and at least one said \emph{offensive}. No trend line is fitted: Spearman's $\rho = +0.236$ ($p = 0.511$). Islam (44.8\%) and Arab (47.9\%) are both above average in contestedness, but Hispanic sits at 44.5\% with a hate-share 9.4 points higher than Islam's. Companion to the slur-content figure; neither confound accounts for the gradient.}
```
