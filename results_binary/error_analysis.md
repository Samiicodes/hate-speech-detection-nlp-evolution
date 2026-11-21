# Error analysis: what RoBERTa's Islam-targeted false positives actually are

Every non-hateful, Islam-targeted post in the HateXplain test split that
`roberta_base` (seed 42) flags as hate. **48 posts** out of a
pool of 137 -- the full population, not a sample.

Slurs are masked. Posts are quoted verbatim otherwise, including the
original spelling and the absence of punctuation.

## Coding scheme

- **`slur_dehumanisation`** — Anti-Muslim slur, or a dehumanising generalisation about Muslims as a group (vermin/savages/animal framing, collective accusations of rape or terror).
- **`nativist_political`** — Hostile political argument about Muslims -- deportation, invasion, demographic replacement, banning Islam -- carrying no slur and no explicit dehumanisation.
- **`doctrinal_attack`** — Attack framed on Islam as doctrine or on Muhammad, rather than directly on Muslims as people.
- **`other_group_primary`** — Hate whose primary target is another group (most often antisemitic), with Muslims mentioned in passing.
- **`ingroup_intraminority`** — Speaker identifies as Muslim, or as a member of another minority speaking about Muslims.
- **`counter_speech`** — Naming, criticising or reporting anti-Muslim treatment -- the failure mode where a model silences the person describing abuse rather than the person committing it.

## Distribution

| category | n | % of flagged |
|---|---|---|
| `slur_dehumanisation` | 25 | 52.1% |
| `nativist_political` | 12 | 25.0% |
| `doctrinal_attack` | 2 | 4.2% |
| `other_group_primary` | 6 | 12.5% |
| `ingroup_intraminority` | 2 | 4.2% |
| `counter_speech` | 1 | 2.1% |
| **total** | **48** | |

Posts where the model silences someone describing or experiencing abuse rather than committing it: **3 of 48 (6.2%)**.

## By original three-class label

| category | normal | offensive |
|---|---|---|
| `slur_dehumanisation` | 1 | 24 |
| `nativist_political` | 1 | 11 |
| `doctrinal_attack` | 0 | 2 |
| `other_group_primary` | 0 | 6 |
| `ingroup_intraminority` | 1 | 1 |
| `counter_speech` | 1 | 0 |
| **total** | **4** | **44** |

## Examples

### `slur_dehumanisation` (25)

> so if the ghetto clans and angry dumb fems so come over with there muslum terrorist buddies that are equally great appearetly for whatever reason will this site cater to them because they do love a good upsetting it like a happening for them except is it different

*10592982_gab · annotators: offensive · rule_based_lexicon=1, tfidf_linear_svc=1, tfidf_logistic_regression=1*

> <user> shitty r******

*1178883172693491712_twitter · annotators: offensive · rule_based_lexicon=1, tfidf_linear_svc=1, tfidf_logistic_regression=1*

> the only way you can tell these sand n****** apart is by the size of the ashtrays on their heads

*12160171_gab · annotators: offensive · rule_based_lexicon=1, tfidf_linear_svc=1, tfidf_logistic_regression=1*

### `nativist_political` (12)

> okay now get the islame ist muslimes out of our country for christ sake training camps really

*10924439_gab · annotators: offensive · rule_based_lexicon=0, tfidf_linear_svc=1, tfidf_logistic_regression=1*

> france problems are not going to fix themselves and this attack was just the tip of the iceberg uncontrolled immigration has been an ongoing issue in france for over a decade the only longstanding solution would be to deport every last moslem back to their home countries mega

*12290514_gab · annotators: offensive · rule_based_lexicon=1, tfidf_linear_svc=1, tfidf_logistic_regression=1*

> <user> <user> every single time nothing an islamist says can ever be trusted

*1270628271265329153_twitter · annotators: offensive · rule_based_lexicon=0, tfidf_linear_svc=0, tfidf_logistic_regression=0*

### `doctrinal_attack` (2)

> yeah you better show alittle more than what islam promotes and does dont see hordes of jews burning christians and preforming honor killings

*19029646_gab · annotators: offensive · rule_based_lexicon=0, tfidf_linear_svc=1, tfidf_logistic_regression=1*

> as an infidel under islam you have three choices convert pay jizya or be killed as an infidel women you get the extra bonus can be raped enslaved bought and sold with islams blessing

*23173358_gab · annotators: offensive · rule_based_lexicon=1, tfidf_linear_svc=0, tfidf_logistic_regression=0*

### `other_group_primary` (6)

> oh we got somebody trying to defend the murderous n**** community and the drug dealing hispanic community you are also defending those mongrels the islam they say white people are all racist and we can only be racist then their stereotype apply as well

*11619975_gab · annotators: offensive · rule_based_lexicon=1, tfidf_linear_svc=0, tfidf_logistic_regression=0*

> swedish prime minister stefan löven condemns minor attack by muslims against the synagogue in gothenburg he wants more money for holocaust propaganda and harsh punishment for antisemitism swedes have been murdered and raped for years without him saying a word a shekel for a good goy

*16001308_gab · annotators: offensive · rule_based_lexicon=1, tfidf_linear_svc=0, tfidf_logistic_regression=0*

> clueless boomer jews dont want refugees because muslims hate them

*17221716_gab · annotators: offensive · rule_based_lexicon=0, tfidf_linear_svc=1, tfidf_logistic_regression=1*

### `ingroup_intraminority` (2)

> <user> its totly gcc countries muslim leaders fault beacuse of them these mother fukers can do anything with muslims of end ia

*1253446187509784576_twitter · annotators: normal · rule_based_lexicon=0, tfidf_linear_svc=1, tfidf_logistic_regression=1*

> they are not my fuxking people b**** i am a hindu not a fuxking sullah muslim us hindus have been dealing with these sullahs longer than your pasty white assess so shut the fuck up

*15997237_gab · annotators: offensive · rule_based_lexicon=1, tfidf_linear_svc=0, tfidf_logistic_regression=0*

### `counter_speech` (1)

> if a muslim person kills innocent people they are considered a terrorist but if a white person does it they are considered mentally ill

*1106567747369086976_twitter · annotators: normal · rule_based_lexicon=1, tfidf_linear_svc=0, tfidf_logistic_regression=0*

## Does the picture shift by paradigm?

The SVM flags 38 posts from the same pool; RoBERTa flags 48. 30 are flagged by both.

| category | both | RoBERTa only | SVM only |
|---|---|---|---|
| `slur_dehumanisation` | 20 | 5 | 2 |
| `nativist_political` | 6 | 6 | 3 |
| `doctrinal_attack` | 1 | 1 | 0 |
| `other_group_primary` | 2 | 4 | 1 |
| `ingroup_intraminority` | 1 | 1 | 1 |
| `counter_speech` | 0 | 1 | 1 |
| **total** | **30** | **18** | **8** |

Of the 8 posts only the SVM flags, **2** fall in the two concerning categories (25%), against 3 of 48 (6%) across all of RoBERTa's.

### Posts only the SVM flags

> we were not paid we were threatened by the globalists and their calls for one worldoligarchy by the blacks and their home invasions and muggings by muslims and their terror and by the commies and their calls for white genocide who pays you

*13749051_gab · annotators: offensive · `other_group_primary`*

> if every muslim was to work and learn english instead of faking back pain for dole put britain first the problems would be drastically smaller coming from a british muslim britainfirst britfam

*1642356_gab · annotators: offensive · `ingroup_intraminority`*

> and naz shah mp liked a comment that told raped white kids by p*** muslims to keep their mouths shut for diversity ad corbyn didnt sack her tories ignored it all

*20217365_gab · annotators: normal · `slur_dehumanisation`*

> are they 2 nd 3 rd 4 th wives the muslim child birth rate is out of control and the <number> bastard are letting it carry on they really believe that soft politics will curb their desire to have a calliohate in the uk <number> mps and so delusional and dangerous

*21333143_gab · annotators: offensive · `nativist_political`*

> very interesting to see what strikes fear into the hearts of leftists it not islamic jihadists it not muslim rape gangs it not ghetto crime it not mexican drug cartels it the idea that a white person could be violent that scares them

*22190329_gab · annotators: offensive · `nativist_political`*

> <number> taxi drivers sentenced and <number> more suspended in newcastle for raping and grooming children why is not this headline news bbc i will tell you it is because they are all pakistani heritage muslims who single out white children to groom and rape ignored by our liberal elite

*25535259_gab · annotators: normal · `nativist_political`*

> i noticed some people talking about valerie jarret being white fact most arabs are caucasian also the name iran means land of aryans fact check me eta it still does not detract from the fact that she a muslim c*** from the planet of the apes

*26596423_gab · annotators: offensive · `slur_dehumanisation`*

> <user> you used the term arab to describe all moslems i pointed out not all arabs are moslems and not all moslems are arabs and you got butthurt runalongsnowflake

*3009506_gab · annotators: normal · `counter_speech`*
