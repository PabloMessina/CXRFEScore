"""Unit tests for text utilities (no models / network)."""

import pytest

from cxrfescore.text_utils import parse_facts, remove_consecutive_repeated_words_from_text


def test_parse_facts_basic():
    txt = '["small pleural effusion", "normal heart size"]'
    assert parse_facts(txt) == ["small pleural effusion", "normal heart size"]


def test_parse_facts_deduplicates():
    txt = '["a", "b", "a"]'
    assert parse_facts(txt) == ["a", "b"]


def test_parse_facts_missing_closing_bracket():
    txt = '["fact one", "fact two"'
    assert parse_facts(txt) == ["fact one", "fact two"]


def test_parse_facts_no_list_returns_empty():
    assert parse_facts("no facts here") == []


def test_remove_consecutive_repeated_words():
    assert (
        remove_consecutive_repeated_words_from_text("effusion effusion present")
        == "effusion present"
    )


def test_remove_consecutive_repeated_phrases():
    text = "right pleural effusion right pleural effusion noted"
    assert (
        remove_consecutive_repeated_words_from_text(text)
        == "right pleural effusion noted"
    )


def test_remove_repeated_words_ks_int():
    assert remove_consecutive_repeated_words_from_text("a a b", ks=1) == "a b"
