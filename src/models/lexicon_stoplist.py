from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).parent

FUNCTION_WORDS: frozenset[str] = frozenset(
    """
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can cannot could couldn't did didn't
do does doesn't doing don't down during each few for from further had hadn't has
hasn't have haven't having he he'd he'll he's her here here's hers herself him
himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself
let's me more most mustn't my myself no nor not of off on once only or other
ought our ours ourselves out over own same shan't she she'd she'll she's should
shouldn't so some such than that that's the their theirs them themselves then
there there's these they they'd they'll they're they've this those through to too
under until up very was wasn't we we'd we'll we're we've were weren't what what's
when when's where where's which while who who's whom why why's with won't would
wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split()
    # Twitter / HateXplain preprocessing artefacts and common contractions
    + """
rt url user number percent amp im dont doesnt cant wont ive youre thats gonna
wanna aint yeah yea ok okay lol lmao haha oh uh hey ya wasn didn couldn shouldn
""".split()
)

IDENTITY_TERMS: frozenset[str] = frozenset(
    """
muslim muslims islam islamic islamist moslem moslems arab arabs arabic pakistani
pakistanis turk turks somali somalis
jew jews jewish judaism israeli israelis zionist zionists
christian christians christianity catholic catholics hindu hindus sikh sikhs
buddhist buddhists atheist atheists
black blacks negroid african africans afro asian asians chinese japanese korean
indian indians latino latinos latina hispanic mexican mexicans white whites
caucasian european europeans
immigrant immigrants migrant migrants refugee refugees foreigner foreigners
gay gays lesbian lesbians bisexual trans transgender transsexual queerfolk
homosexuals lgbt
woman women female females man men male males
disabled disability autistic
""".split()
)


def _load_innocuous(path: Path | None = None) -> frozenset[str]:
    path = path or (_HERE / "innocuous_stoplist.txt")
    if not path.exists():
        return frozenset()
    words = (
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )
    return frozenset(words)


INNOCUOUS_TERMS: frozenset[str] = _load_innocuous()

# Everything struck before the top-K cut is applied.
FULL_STOPLIST: frozenset[str] = (
    FUNCTION_WORDS | IDENTITY_TERMS | INNOCUOUS_TERMS
)

__all__ = [
    "FULL_STOPLIST",
    "FUNCTION_WORDS",
    "IDENTITY_TERMS",
    "INNOCUOUS_TERMS",
]
