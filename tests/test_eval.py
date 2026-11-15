"""
Unit tests for self-generated benchmark dataset and evaluation harness.
"""

import pytest
import json
from src.config import MY_QUESTIONS_FILE
from src.verify import verify_citation


def test_my_questions_dataset_format():
    assert MY_QUESTIONS_FILE.exists()
    with open(MY_QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    assert len(questions) == 10
    required_keys = {"question_id", "team", "question", "ground_truth_answer", "citations"}
    for q in questions:
        assert required_keys.issubset(q.keys())
        assert 1 <= len(q["citations"]) <= 3


def test_my_questions_citations_100_percent_verbatim():
    with open(MY_QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    for q in questions:
        for c in q["citations"]:
            res = verify_citation(c["meeting_id"], c["quote"], snap=False)
            assert res.valid is True, f"Failed citation in {q['question_id']}: {c}"
            assert res.word_count >= 6
