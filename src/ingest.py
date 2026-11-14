
# determined ingestion and SQLite FTS5 indexing for AMI Meeting Transcripts

# Transforms raw transcript files into:
# meetings table - meeting level meta data
# turns table - Structured speaker turns with timestamps and sequence numbers
# turns_fts virtual table - SQLite FTS5 full text index for fast BM25 search


from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from src.config import (
    TRANSCRIPTS_DIR,
    DB_PATH,
    STORAGE_DIR,
    ensure_dirs,
)

# Regex patterns for deterministic parsing
HEADER_MEETING_RE = re.compile(r"^Meeting:\s*([A-Za-z0-9]+)", re.IGNORECASE)
HEADER_TEAM_RE = re.compile(r"^Team:\s*([A-Za-z0-9]+)", re.IGNORECASE)
HEADER_NUMBER_RE = re.compile(r"^Meeting number:\s*(\d+)\s*\(([a-z])\)", re.IGNORECASE)
HEADER_SPEAKERS_RE = re.compile(r"^Speakers:\s*([^\n]+)", re.IGNORECASE)

TURN_HEADER_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\s+([A-Za-z0-9 _]+)$")


@dataclass
class TurnRecord:
    meeting_id: str
    team: str
    session: str
    session_seq: int
    turn_id: int
    timestamp: str
    seconds: int
    speaker: str
    text: str
    word_count: int


@dataclass
class MeetingRecord:
    meeting_id: str
    team: str
    session: str
    session_seq: int
    speakers: str
    total_turns: int
    total_words: int
    duration_sec: int


def get_db_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return an SQLite connection with sqlite3.Row factory and performance pragmas."""
    path = db_path or DB_PATH
    ensure_dirs()
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize SQLite relational tables and FTS5 virtual table."""
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS meetings (
                meeting_id TEXT PRIMARY KEY,
                team TEXT NOT NULL,
                session TEXT NOT NULL,
                session_seq INTEGER NOT NULL,
                speakers TEXT,
                total_turns INTEGER NOT NULL,
                total_words INTEGER NOT NULL,
                duration_sec INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id TEXT NOT NULL,
                team TEXT NOT NULL,
                session TEXT NOT NULL,
                session_seq INTEGER NOT NULL,
                turn_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                seconds INTEGER NOT NULL,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                FOREIGN KEY (meeting_id) REFERENCES meetings(meeting_id)
            );

            CREATE INDEX IF NOT EXISTS idx_turns_team_seq ON turns(team, session_seq, turn_id);
            CREATE INDEX IF NOT EXISTS idx_turns_meeting_turn ON turns(meeting_id, turn_id);

            -- SQLite FTS5 Full-Text Search Virtual Table
            CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
                text,
                meeting_id UNINDEXED,
                team UNINDEXED,
                speaker UNINDEXED,
                turn_id UNINDEXED,
                tokenize = 'porter unicode61'
            );
        """)


def parse_transcript_file(file_path: Path) -> Tuple[MeetingRecord, List[TurnRecord]]:
# Parse a single AMI meeting transcript file deterministically
    filename_stem = file_path.stem
    meeting_id = filename_stem
    # Derive team and session from filename (e.g. TS3007b -> team TS3007, session b)
    team_match = re.match(r"^([A-Za-z0-9]+?)([a-d])$", filename_stem, re.IGNORECASE)
    if team_match:
        team = team_match.group(1).upper()
        session = team_match.group(2).lower()
    else:
        team = filename_stem
        session = "a"

    session_seq_map = {"a": 1, "b": 2, "c": 3, "d": 4}
    session_seq = session_seq_map.get(session, 1)
    speakers = ""

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    turns: List[TurnRecord] = []
    turn_counter = 0
    current_timestamp = ""
    current_seconds = 0
    current_speaker = ""
    current_text_parts: List[str] = []
    in_turns_body = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        # Check for turn header (HH:MM:SS Speaker X)
        turn_match = TURN_HEADER_RE.match(line)
        if turn_match:
            in_turns_body = True
            # Flush previous turn if exists
            if current_speaker and current_text_parts:
                turn_counter += 1
                text = " ".join(current_text_parts).strip()
                turns.append(TurnRecord(
                    meeting_id=meeting_id,
                    team=team,
                    session=session,
                    session_seq=session_seq,
                    turn_id=turn_counter,
                    timestamp=current_timestamp,
                    seconds=current_seconds,
                    speaker=current_speaker,
                    text=text,
                    word_count=len(text.split()),
                ))
                current_text_parts = []

            hh, mm, ss, spk = turn_match.groups()
            current_timestamp = f"{hh}:{mm}:{ss}"
            current_seconds = int(hh) * 3600 + int(mm) * 60 + int(ss)
            current_speaker = spk.strip()
            continue

        if in_turns_body:
            current_text_parts.append(line)
        else:
            # Parse header metadata lines
            m_id = HEADER_MEETING_RE.match(line)
            if m_id:
                meeting_id = m_id.group(1).strip()
            m_team = HEADER_TEAM_RE.match(line)
            if m_team:
                team = m_team.group(1).strip().upper()
            m_num = HEADER_NUMBER_RE.match(line)
            if m_num:
                session_seq = int(m_num.group(1))
                session = m_num.group(2).lower()
            m_spk = HEADER_SPEAKERS_RE.match(line)
            if m_spk:
                speakers = m_spk.group(1).strip()

    # Flush final turn
    if current_speaker and current_text_parts:
        turn_counter += 1
        text = " ".join(current_text_parts).strip()
        turns.append(TurnRecord(
            meeting_id=meeting_id,
            team=team,
            session=session,
            session_seq=session_seq,
            turn_id=turn_counter,
            timestamp=current_timestamp,
            seconds=current_seconds,
            speaker=current_speaker,
            text=text,
            word_count=len(text.split()),
        ))

    total_words = sum(t.word_count for t in turns)
    duration_sec = turns[-1].seconds if turns else 0

    meeting_record = MeetingRecord(
        meeting_id=meeting_id,
        team=team,
        session=session,
        session_seq=session_seq,
        speakers=speakers,
        total_turns=len(turns),
        total_words=total_words,
        duration_sec=duration_sec,
    )

    return meeting_record, turns


def ingest_all_transcripts(force: bool = False, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Ingest all 138 transcript files into SQLite and FTS5.
    If database already contains transcripts and force=False, returns existing counts.
    """
    conn = get_db_connection(db_path)
    init_db(conn)

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM meetings")
    existing_meetings = cursor.fetchone()[0]

    if existing_meetings > 0 and not force:
        cursor.execute("SELECT COUNT(*) FROM turns")
        existing_turns = cursor.fetchone()[0]
        return {
            "status": "already_indexed",
            "meetings_count": existing_meetings,
            "turns_count": existing_turns,
        }

    start_time = time.time()
    transcript_files = sorted(list(TRANSCRIPTS_DIR.glob("*.txt")))
    if not transcript_files:
        raise FileNotFoundError(f"No transcript files found in {TRANSCRIPTS_DIR}")

    all_meetings: List[MeetingRecord] = []
    all_turns: List[TurnRecord] = []

    for file_path in transcript_files:
        m_rec, t_recs = parse_transcript_file(file_path)
        all_meetings.append(m_rec)
        all_turns.extend(t_recs)

    with conn:
        # Clear existing rows if re-indexing
        conn.execute("DELETE FROM meetings")
        conn.execute("DELETE FROM turns")
        conn.execute("DELETE FROM turns_fts")

        conn.executemany("""
            INSERT INTO meetings (
                meeting_id, team, session, session_seq, speakers,
                total_turns, total_words, duration_sec
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (m.meeting_id, m.team, m.session, m.session_seq, m.speakers,
             m.total_turns, m.total_words, m.duration_sec)
            for m in all_meetings
        ])

        conn.executemany("""
            INSERT INTO turns (
                meeting_id, team, session, session_seq, turn_id,
                timestamp, seconds, speaker, text, word_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (t.meeting_id, t.team, t.session, t.session_seq, t.turn_id,
             t.timestamp, t.seconds, t.speaker, t.text, t.word_count)
            for t in all_turns
        ])

        conn.executemany("""
            INSERT INTO turns_fts (
                text, meeting_id, team, speaker, turn_id
            ) VALUES (?, ?, ?, ?, ?)
        """, [
            (t.text, t.meeting_id, t.team, t.speaker, t.turn_id)
            for t in all_turns
        ])

    elapsed = time.time() - start_time
    total_words = sum(m.total_words for m in all_meetings)

    return {
        "status": "indexed_fresh",
        "meetings_count": len(all_meetings),
        "turns_count": len(all_turns),
        "words_count": total_words,
        "elapsed_sec": round(elapsed, 3),
        "db_path": str(db_path or DB_PATH),
    }

# Database Query Helpers

def search_turns(
    query: str,
    team: Optional[str] = None,
    meeting_id: Optional[str] = None,
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Perform BM25-ranked full-text search using SQLite FTS5.
    Optionally filters by team or specific meeting_id.
    """
    conn = get_db_connection(db_path)
    
    # Sanitize query for FTS5 syntax
    # Strip dangerous characters while preserving terms
    clean_terms = [t for t in re.findall(r"\w+", query) if t]
    if not clean_terms:
        return []
    
    fts_query = " OR ".join(clean_terms)

    conditions = ["turns_fts MATCH ?"]
    params: List[Any] = [fts_query]

    if team:
        conditions.append("t.team = ?")
        params.append(team.upper())
    if meeting_id:
        conditions.append("t.meeting_id = ?")
        params.append(meeting_id)

    where_clause = " AND ".join(conditions)
    params.append(limit)

    sql = f"""
        SELECT 
            t.meeting_id,
            t.team,
            t.session,
            t.session_seq,
            t.turn_id,
            t.timestamp,
            t.speaker,
            t.text,
            bm25(turns_fts) AS rank
        FROM turns_fts
        JOIN turns t ON t.meeting_id = turns_fts.meeting_id AND t.turn_id = turns_fts.turn_id
        WHERE {where_clause}
        ORDER BY rank
        LIMIT ?
    """

    cursor = conn.cursor()
    rows = cursor.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_team_turns(
    team: str,
    max_session: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch all chronological turns for a team across all sessions (a -> b -> c -> d).
    Optionally filter up to a maximum session (e.g. max_session='c' for pre-final brief).
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    sql = """
        SELECT meeting_id, team, session, session_seq, turn_id, timestamp, seconds, speaker, text
        FROM turns
        WHERE team = ?
    """
    params: List[Any] = [team.upper()]

    if max_session:
        session_seq_map = {"a": 1, "b": 2, "c": 3, "d": 4}
        max_seq = session_seq_map.get(max_session.lower(), 4)
        sql += " AND session_seq <= ?"
        params.append(max_seq)

    sql += " ORDER BY session_seq ASC, turn_id ASC"
    
    rows = cursor.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_turn(meeting_id: str, turn_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Fetch a specific speaker turn by meeting_id and turn_id."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    row = cursor.execute(
        "SELECT * FROM turns WHERE meeting_id = ? AND turn_id = ?",
        (meeting_id, turn_id)
    ).fetchone()
    return dict(row) if row else None


def get_turn_context(
    meeting_id: str,
    turn_id: int,
    window: int = 2,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Fetch a target turn along with surrounding turns for conversational context."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    rows = cursor.execute(
        """
        SELECT meeting_id, turn_id, timestamp, speaker, text
        FROM turns
        WHERE meeting_id = ? AND turn_id BETWEEN ? AND ?
        ORDER BY turn_id ASC
        """,
        (meeting_id, max(1, turn_id - window), turn_id + window)
    ).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    from rich import print as rprint
    rprint("[bold blue]Starting AMI Transcripts Ingestion...[/bold blue]")
    result = ingest_all_transcripts(force=True)
    rprint(f"[bold green]Ingestion Complete in {result['elapsed_sec']} seconds![/bold green]")
    rprint(f"  Meetings Indexed: [cyan]{result['meetings_count']}[/cyan]")
    rprint(f"  Turns Indexed:    [cyan]{result['turns_count']}[/cyan]")
    rprint(f"  Total Words:      [cyan]{result['words_count']:,}[/cyan]")
    rprint(f"  Database Path:    [dim]{result['db_path']}[/dim]")

    # Run a test query
    rprint("\n[bold yellow]Testing FTS5 Search for Team TS3007 (Query: 'battery kinetic'):[/bold yellow]")
    hits = search_turns("battery kinetic", team="TS3007", limit=3)
    for h in hits:
        rprint(f"  [{h['meeting_id']} | Turn {h['turn_id']} | {h['speaker']} | Rank {h['rank']:.3f}]: {h['text'][:90]}...")
