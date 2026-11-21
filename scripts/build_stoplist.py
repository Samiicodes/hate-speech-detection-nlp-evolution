from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from src import data
from src.models.lexicon_stoplist import FUNCTION_WORDS, IDENTITY_TERMS

TOKEN_RE = re.compile(r"[a-z']+")

# Hate-associated tokens
HATE_ASSOCIATED: frozenset[str] = frozenset("""
nigger niggers nigga niggas kike kikes muzzie muzzies muzrat raghead spic paki
negro chong ching shitskin mudshark cuck faggot faggots fag fags queers queer
dyke dykes retard retards retarded homosexual hillbilly redneck slut sluts hoe
hoes bitch bitches whore whores cunt cunts pussy dick dicks ass asshole assholes
idiot idiots trash alien aliens ghetto coon coons wetback tranny trannies gook
chink chinks beaner savage savages moron morons loser losers scumbag scumbags
freak freaks creep creeps
fuck fucking fucked fuckin fucks shit shits shithole bullshit damn wtf lmfao
suck sucks piss pissed crap
kill kills killed killing die dies died dead death murder murders murdered rape
rapes raped raping rapist rapists violence violent genocide beat beating destroy
destroyed attack attacks shoot shot shooting gun guns jail fight fighting war
blood hang hanged holocaust nazi nazis hitler supremacy supremacist supremacists
gas gassed exterminate slaughter lynch bomb bombs burn burned drown stab
filthy filth dirty disgusting disgust gross nasty scum ugly stupid stupidity
dumb sick sickening evil fat crazy insane mad lying lies liar fake mentally
worthless pathetic vile subhuman degenerate parasite parasites vermin animals
animal rats rat cockroach cockroaches inbred backward primitive barbaric
sharia jihadi jihad terrorist terrorists terror terrorism immigration immigrant
illegal illegals welfare antifa maga liberals liberal leftist leftists communist
communists racism racist racists homophobic propaganda censored diversity deport
deported invader invaders invasion
shut hell blame harassment banned ban forced deserve deserves victims wrong
sand alt wing wall gab trailer mass gang gangs crime criminal criminals thug
thugs low skin brown color loving send piece
""".split())

HEADER = """\
# Innocuous high-frequency tokens
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-freq", type=int, default=20)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src" / "models" / "innocuous_stoplist.txt",
    )
    args = parser.parse_args()

    train, _, _ = data.get_dataset()

    doc_freq: Counter = Counter()
    for text in train["text"]:
        doc_freq.update(set(TOKEN_RE.findall(text.lower())))

    candidates = {
        w for w, c in doc_freq.items()
        if c >= args.min_freq and len(w) >= args.min_len
    }
    struck = sorted(candidates - HATE_ASSOCIATED - FUNCTION_WORDS - IDENTITY_TERMS)
    kept = sorted(candidates & HATE_ASSOCIATED)

    header = HEADER.format(
        min_freq=args.min_freq, n_cand=len(candidates),
        n_struck=len(struck), n_kept=len(kept),
    )
    args.out.write_text(header + "\n".join(struck) + "\n", encoding="utf-8")

    print(f"candidates : {len(candidates)}")
    print(f"struck     : {len(struck)}  -> {args.out}")
    print(f"kept       : {len(kept)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
