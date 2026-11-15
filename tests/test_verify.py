"""
Unit tests for verbatim citation verification and auto-snapping.
"""

import pytest
from src.verify import (
    verify_citation,
    verify_and_repair_citations,
    count_words,
)


def test_count_words():
    assert count_words("one two three four five six") == 6
    assert count_words("  hello   world  ") == 2
    assert count_words("") == 0


def test_exact_verbatim_citation():
    # TS3007b Turn 145 / Turn 46
    quote = "Uh first you have the energy source."
    res = verify_citation("TS3007b", quote, snap=False)
    assert res.valid is True
    assert res.snapped is False
    assert res.word_count >= 6
    assert res.turn_id is not None
    assert res.speaker == "Speaker C"


def test_citation_too_short():
    quote = "hello world"
    res = verify_citation("TS3007b", quote, snap=True)
    assert res.valid is False
    assert "minimum required is 6" in res.reason


def test_auto_snapping_punctuation_and_caps():
    # Imperfect quote with lowercase and missing punctuation
    imperfect = "first you have the energy source The energy source would be a battery"
    res = verify_citation("TS3007b", imperfect, snap=True)
    assert res.valid is True
    assert res.snapped is True
    assert res.similarity_score > 90.0
    # Must be an exact substring in the database turn now
    exact_res = verify_citation("TS3007b", res.quote, snap=False)
    assert exact_res.valid is True


def test_hallucinated_quote_fails():
    fake_quote = "we definitely decided to build a nuclear powered flying car with wings"
    res = verify_citation("TS3007b", fake_quote, snap=True)
    assert res.valid is False
    assert "Quote not found verbatim" in res.reason


def test_verify_and_repair_citations_batch():
    candidate_citations = [
        # 1. Valid exact
        {"meeting_id": "TS3007b", "quote": "Uh first you have the energy source."},
        # 2. Valid imperfect (needs snap)
        {"meeting_id": "TS3007b", "quote": "first you have the energy source The energy source would be a battery"},
        # 3. Invalid hallucinated
        {"meeting_id": "TS3007b", "quote": "this quote is completely made up by an imaginary agent"},
        # 4. Invalid too short
        {"meeting_id": "TS3007b", "quote": "too short"},
    ]

    repaired, reports = verify_and_repair_citations(candidate_citations, snap=True, max_citations=3)

    # Should retain the 2 valid/repaired citations and discard the invalid ones
    assert len(repaired) == 2
    assert len(reports) == 4
    assert reports[0].valid is True
    assert reports[1].valid is True
    assert reports[2].valid is False
    assert reports[3].valid is False
