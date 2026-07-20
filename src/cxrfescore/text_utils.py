"""Helpers for parsing and cleaning extracted radiology facts."""

import json
import re

_COMMA_SEPARATED_LIST_REGEX = re.compile(r'\[\s*(\".+?\"(\s*,\s*\".+?\")*)?\s*\]?')


def parse_facts(txt: str) -> list:
    """Parse a JSON-like list of facts from T5 fact-extractor output."""
    match = _COMMA_SEPARATED_LIST_REGEX.search(txt)
    if match is None:
        return []
    facts_str = match.group()
    if facts_str[-1] != "]":
        facts_str += "]"
    facts = json.loads(facts_str)
    seen = set()
    clean_facts = []
    for fact in facts:
        if fact not in seen:
            clean_facts.append(fact)
            seen.add(fact)
    return clean_facts


def _substrings_are_equal(text, i, j, k):
    for x in range(k):
        if text[i + x] != text[j + x]:
            return False
    return True


def remove_consecutive_repeated_words_from_text(text, ks=None):
    """Remove consecutive repeated word n-grams from text."""
    if ks is None:
        ks = [1, 2, 3, 4, 5, 6, 7, 8]
    assert type(ks) is int or type(ks) is list
    if type(ks) is int:
        ks = [ks]
    else:
        assert len(ks) > 0
        assert all(type(x) is int for x in ks)

    tokens = text.split()
    lower_tokens = text.lower().split()
    dedup_tokens = []
    dedup_lower_tokens = []

    for k in ks:
        for i in range(len(lower_tokens)):
            skip = False
            for j in range(k):
                s = i - j
                e = s + k - 1
                if (
                    s - k >= 0
                    and e < len(lower_tokens)
                    and _substrings_are_equal(lower_tokens, s, s - k, k)
                ):
                    skip = True
                    break
            if skip:
                continue
            dedup_tokens.append(tokens[i])
            dedup_lower_tokens.append(lower_tokens[i])
        tokens = dedup_tokens
        lower_tokens = dedup_lower_tokens
        dedup_tokens = []
        dedup_lower_tokens = []
    return " ".join(tokens)
