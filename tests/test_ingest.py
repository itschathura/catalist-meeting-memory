# Unit tests for ingestion and FTs5 search

import pytest
import sqlite3
from pathlib import Path
from src.config import TRANSCRIPTS_DIR, DB_PATH
from src.ingest import (
    parse_transcript_file,
    get_db_connection,
    init_db,
    ingest_all_transcripts,
    search_turns,
    get_turn,
    get_team_turns,
    get_turn_context,
)


def test_parse_single_transcript():
    sample_file = TRANSCRIPTS_DIR / "TS3007b.txt"
    assert sample_file.exists()

    m_rec, turns = parse_transcript_file(sample_file)
    assert m_rec.meeting_id == "TS3007b"
    assert m_rec.team == "TS3007"
    assert m_rec.session == "b"
    assert m_rec.session_seq == 2
    assert len(turns) > 0

    first_turn = turns[0]
    assert first_turn.turn_id == 1
    assert first_turn.speaker == "Speaker A"
    assert "Hello" in first_turn.text


def test_database_populated():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM meetings")
    meetings_count = cursor.fetchone()[0]
    assert meetings_count == 138

    cursor.execute("SELECT COUNT(*) FROM turns")
    turns_count = cursor.fetchone()[0]
    assert turns_count == 48148


def test_fts5_search():
    # Team TS3007 energy source query
    results = search_turns("battery kinetic", team="TS3007", limit=5)
    assert len(results) > 0
    assert all(r["team"] == "TS3007" for r in results)
    assert any("kinetic" in r["text"].lower() or "battery" in r["text"].lower() for r in results)


def test_turn_retrieval_and_context():
    turn = get_turn("TS3007b", 1)
    assert turn is not None
    assert turn["meeting_id"] == "TS3007b"
    assert turn["turn_id"] == 1

    context = get_turn_context("TS3007b", 2, window=1)
    # Should fetch turn 1, 2, 3
    assert len(context) == 3
    turn_ids = [t["turn_id"] for t in context]
    assert turn_ids == [1, 2, 3]


def test_chronological_team_turns():
    turns = get_team_turns("TS3007", max_session="c")
    assert len(turns) > 0
    sessions = set(t["session"] for t in turns)
    assert "d" not in sessions
    assert {"a", "b", "c"}.issubset(sessions)
