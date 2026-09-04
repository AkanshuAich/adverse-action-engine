"""Flesch reading ease, implemented directly.

Plain language is a regulatory expectation, not a nicety: a notice that states
the reasons in prose an applicant cannot follow has not really stated them.

Implemented here rather than taken from a library because the only library
option pulls nltk, which currently carries PYSEC-2026-3740 with no fix
released. The formula is three terms and a syllable count, so taking a
vulnerable transitive dependency for one number would be a poor trade.

The syllable count is a heuristic - vowel groups, with the usual corrections
for silent trailing "e" and for "-le" endings. It is wrong on some words, as
every implementation of this metric is, including the ones in libraries. That
matters less than it looks: the score is used to compare notices against each
other and against a baseline, so a consistent bias cancels out. It is not used
to make a claim about any individual letter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

_SENTENCE_END: Final[re.Pattern[str]] = re.compile(r"[.!?]+")
_WORD: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z'-]*")
_VOWEL_GROUP: Final[re.Pattern[str]] = re.compile(r"[aeiouy]+")

# Flesch reading ease: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
_BASE: Final[float] = 206.835
_WORDS_PER_SENTENCE_WEIGHT: Final[float] = 1.015
_SYLLABLES_PER_WORD_WEIGHT: Final[float] = 84.6

PLAIN_LANGUAGE_FLOOR: Final[float] = 40.0
"""Below this a notice reads as dense administrative prose.

Not a legal threshold - no regulator specifies a Flesch score. It is a
tripwire for the eval report, chosen so that ordinary bank letter-writing
passes and impenetrable prose does not.
"""


def count_syllables(word: str) -> int:
    """Estimate the syllables in one word.

    Args:
        word: A single word, letters only.

    Returns:
        At least one syllable for any non-empty word.
    """
    lowered = word.lower().strip("'-")
    if not lowered:
        return 0

    groups = _VOWEL_GROUP.findall(lowered)
    count = len(groups)

    # "hope" is one syllable, not two. Skipped for "-le", "-ee" and "-ye",
    # where the trailing e is not silent: "table" and "little" keep both.
    #
    # No separate "-le" adjustment is needed, and adding one is a common bug:
    # the vowel-group pass has already counted that e, so incrementing again
    # gives "table" three syllables. Implementations that do add one strip the
    # silent e first and are putting it back.
    if lowered.endswith("e") and not lowered.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1

    return max(count, 1)


@dataclass(frozen=True)
class ReadabilityScore:
    """A readability measurement and the counts behind it.

    Attributes:
        flesch_reading_ease: Higher is easier. Roughly: 90+ is very easy, 60-70
            plain English, 30-50 difficult, below 30 very difficult.
        words: Word count.
        sentences: Sentence count.
        syllables: Estimated syllable count.
    """

    flesch_reading_ease: float
    words: int
    sentences: int
    syllables: int

    @property
    def is_plain_language(self) -> bool:
        """Whether the text clears the plain-language tripwire."""
        return self.flesch_reading_ease >= PLAIN_LANGUAGE_FLOOR

    @property
    def words_per_sentence(self) -> float:
        """Mean sentence length."""
        return self.words / self.sentences if self.sentences else 0.0


def score_readability(text: str) -> ReadabilityScore:
    """Measure how easily a notice reads.

    Args:
        text: The notice body.

    Returns:
        The score and its component counts. Empty text scores zero rather than
        dividing by zero, and is reported as such.
    """
    words = _WORD.findall(text)
    sentence_count = max(len([part for part in _SENTENCE_END.split(text) if part.strip()]), 1)

    if not words:
        return ReadabilityScore(
            flesch_reading_ease=0.0, words=0, sentences=sentence_count, syllables=0
        )

    syllables = sum(count_syllables(word) for word in words)

    score = (
        _BASE
        - _WORDS_PER_SENTENCE_WEIGHT * (len(words) / sentence_count)
        - _SYLLABLES_PER_WORD_WEIGHT * (syllables / len(words))
    )

    return ReadabilityScore(
        flesch_reading_ease=round(score, 2),
        words=len(words),
        sentences=sentence_count,
        syllables=syllables,
    )
