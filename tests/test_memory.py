"""
Unit tests for Temporal Memory Ledger and Role Resolution.
"""

import pytest
from src.memory import (
    get_team_roles,
    build_temporal_timeline,
    get_pre_final_brief_context,
)


def test_get_team_roles():
    roles = get_team_roles("TS3007")
    assert roles["Speaker A"] == "Project Manager"
    assert roles["Speaker B"] == "User Interface Designer"
    assert roles["Speaker C"] == "Industrial Designer"
    assert roles["Speaker D"] == "Marketing"


def test_build_temporal_timeline_chronological():
    timeline = build_temporal_timeline("TS3007", "battery kinetic")
    assert timeline.team == "TS3007"
    assert len(timeline.all_turns) > 0

    # Ensure strictly sorted chronologically (session_seq, turn_id)
    for i in range(len(timeline.all_turns) - 1):
        t1 = timeline.all_turns[i]
        t2 = timeline.all_turns[i + 1]
        assert (t1.session_seq, t1.turn_id) <= (t2.session_seq, t2.turn_id)


def test_build_temporal_timeline_max_session():
    # Pre-final constraint: only up to session 'c'
    timeline = build_temporal_timeline("TS3012", "speech screen", max_session="c")
    assert "d" not in timeline.turns_by_session
    for t in timeline.all_turns:
        assert t.session in ["a", "b", "c"]
        assert t.session != "d"


def test_timeline_prompt_formatting():
    timeline = build_temporal_timeline("TS3007", "battery kinetic")
    prompt_str = timeline.to_prompt_text()
    assert "CHRONOLOGICAL MEETING TIMELINE" in prompt_str
    assert "Speaker C (Industrial Designer)" in prompt_str
    assert "TS3007b" in prompt_str


def test_pre_final_brief_context():
    context = get_pre_final_brief_context("TS3012")
    assert context["team"] == "TS3012"
    assert context["total_turns"] > 0
    # Must not contain meeting d
    for t in context["turns"]:
        assert t.session in ["a", "b", "c"]
        assert t.session != "d"
