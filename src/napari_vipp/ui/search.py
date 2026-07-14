"""Text normalization and fuzzy matching helpers for UI search fields."""

from __future__ import annotations


def _normalize_search_text(value) -> str:
    return "".join(
        character.lower() if character.isalnum() else " "
        for character in str(value or "")
    ).strip()


def _fuzzy_match(query: str, haystack: str) -> bool:
    tokens = query.split()
    if not tokens:
        return True
    return all(_fuzzy_token_match(token, haystack) for token in tokens)


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


__all__ = [
    "_fuzzy_match",
    "_fuzzy_token_match",
    "_normalize_search_text",
]
