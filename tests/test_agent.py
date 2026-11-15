"""
Unit tests for Retrieval Agent and Pre-Final Brief Generator.
"""

import pytest
from pathlib import Path
from src.agent import answer_question, extract_search_terms
from src.brief import generate_ts3012_brief
from src.config import BRIEF_OUTPUT_FILE


def test_extract_search_terms():
    q = "Team TS3007's energy source changed between meetings. What did they first settle on?"
    terms = extract_search_terms(q)
    assert "energy" in terms
    assert "source" in terms
    assert "between" not in terms
    assert "what" not in terms


def test_answer_question_end_to_end():
    q_text = "Team TS3007's energy source changed between meetings. What did they first settle on, what did they end with, and who proposed the change?"
    result = answer_question("Q05", "TS3007", q_text, use_cache=True)

    assert result["question_id"] == "Q05"
    assert result["team"] == "TS3007"
    assert len(result["answer"]) > 20
    assert 1 <= len(result["citations"]) <= 3

    # Check that citations are verified
    for c in result["citations"]:
        assert len(c["quote"].split()) >= 6
        assert c["meeting_id"].startswith("TS3007")

    # Check verification reports
    assert any(r["valid"] for r in result["verification_reports"])


def test_brief_ts3012_exists_and_structured():
    assert BRIEF_OUTPUT_FILE.exists()
    content = BRIEF_OUTPUT_FILE.read_text(encoding="utf-8")

    # Verify required 4 sections
    assert "1. Decisions Settled So Far" in content
    assert "2. What Changed Along the Way" in content
    assert "3. Open Action Items & Assigned Owners" in content
    assert "4. Open Disagreements & Unresolved Questions" in content

    # Verify no look-ahead leakage into meeting D
    assert "TS3012d" not in content
