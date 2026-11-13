"""Text normalization and fuzzy matching helpers for UI search fields."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise


def _normalize_search_text(value) -> str:
    return "".join(
        character.lower() if character.isalnum() else " "
        for character in str(value or "")
    ).strip()


def _fuzzy_match(query: str, haystack: str, *, aliases: tuple[str, ...] = ()) -> bool:
    """Keep fuzzy name matching, with literal tokens in optional alternatives.

    Do not join aliases into the fuzzy haystack: unrelated letters across a
    growing list of synonyms would otherwise create accidental matches.
    """
    tokens = query.split()
    if not tokens:
        return True
    alternatives = tuple(_normalize_search_text(alias) for alias in aliases)
    return all(
        _fuzzy_token_match(token, haystack)
        or any(token in alias for alias in alternatives)
        for token in tokens
    )


def _fuzzy_token_match(token: str, haystack: str) -> bool:
    if token in haystack:
        return True
    position = 0
    for character in token:
        position = haystack.find(character, position)
        if position < 0:
            return False
        position += 1
    return True


@dataclass(frozen=True)
class NodeLibrarySearchTerms:
    """Index names, aliases and categories without merging their letters."""

    literal_words: tuple[str, ...]
    name_words: tuple[str, ...]
    name_pairs: tuple[tuple[str, str], ...]

    @classmethod
    def from_fields(
        cls,
        title: str,
        operation_id: str,
        *,
        aliases: tuple[str, ...],
        category: str,
        subgroup: str,
    ) -> NodeLibrarySearchTerms:
        names = [
            _normalize_search_text(field).split() for field in (title, operation_id)
        ]
        name_words = tuple(dict.fromkeys(word for name in names for word in name))
        name_pairs = tuple(
            dict.fromkeys(pair for name in names for pair in pairwise(name))
        )
        other_words = (
            word
            for field in (category, subgroup, *aliases)
            for word in _normalize_search_text(field).split()
        )
        literal_words = tuple(dict.fromkeys((*name_words, *other_words)))
        return cls(literal_words, name_words, name_pairs)

    def match_priority(self, tokens: tuple[str, ...]) -> int | None:
        """Prefer direct word prefixes, then one typo, then short name shorthand."""
        unmatched = [
            token
            for token in tokens
            if not any(word.startswith(token) for word in self.literal_words)
        ]
        if not unmatched:
            return 0
        if len(unmatched) != 1 or len(unmatched[0]) < 4:
            return None
        token = unmatched[0]
        if any(_one_typo(token, word) for word in self.name_words):
            return 1
        if any(
            token[0] == first[0] and _abbreviates_prefix(token[1:], second)
            for first, second in self.name_pairs
        ):
            return 2
        return None


def _one_typo(token: str, word: str) -> bool:
    """A single insertion, deletion, substitution, or adjacent transposition."""
    if abs(len(token) - len(word)) > 1:
        return False
    index = next(
        (i for i, (a, b) in enumerate(zip(token, word, strict=False)) if a != b),
        min(len(token), len(word)),
    )
    if len(token) == len(word):
        return index < len(token) and (
            token[index + 1 :] == word[index + 1 :]
            or (
                token[index : index + 2] == word[index : index + 2][::-1]
                and token[index + 2 :] == word[index + 2 :]
            )
        )
    short, long = (token, word) if len(token) < len(word) else (word, token)
    return short[index:] == long[index + 1 :]


def _abbreviates_prefix(tail: str, word: str) -> bool:
    """Allow e.g. gblr for Gaussian Blur, not a subsequence of the entire row."""
    if word.startswith(tail):
        return True
    prefix = word[: len(tail) + 1]
    return any(tail == prefix[:i] + prefix[i + 1 :] for i in range(1, len(prefix)))


__all__ = [
    "NodeLibrarySearchTerms",
    "_fuzzy_match",
    "_fuzzy_token_match",
    "_normalize_search_text",
]
