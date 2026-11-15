"""
Verbatim Citation Verifier and Auto-Snapper for AMI Meeting Memory.

Enforces benchmark citation constraints:
1. Citation format: {"meeting_id": str, "quote": str}
2. Exactly 1 to 3 citations per answer.
3. Every quote must be >= 6 consecutive words copied word-for-word from
   WITHIN A SINGLE SPEAKER TURN in that exact meeting transcript.
4. Includes an Auto-Snapper (using RapidFuzz) to align minor LLM discrepancies
   (punctuation/capitalization/filler words) back to exact verbatim transcript substrings.
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from rapidfuzz import fuzz

from src.config import (
    DB_PATH,
    MIN_CITATION_WORDS,
    MAX_CITATIONS_PER_ANSWER,
)
from src.ingest import get_db_connection


@dataclass
class Citation:
    meeting_id: str
    quote: str

    def to_dict(self) -> Dict[str, str]:
        return {"meeting_id": self.meeting_id, "quote": self.quote}


@dataclass
class VerificationResult:
    valid: bool
    meeting_id: str
    quote: str
    original_quote: str
    word_count: int
    turn_id: Optional[int] = None
    speaker: Optional[str] = None
    snapped: bool = False
    similarity_score: float = 100.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def count_words(text: str) -> int:
    """Count words in text by whitespace splitting."""
    return len(text.strip().split())


def find_exact_quote_in_turn(
    meeting_id: str,
    quote: str,
    db_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """
    substring check in SQLITE.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    # Query for exact substring in text column for the specified meeting
    sql = """
        SELECT turn_id, speaker, text
        FROM turns
        WHERE meeting_id = ? AND instr(text, ?) > 0
        LIMIT 1
    """
    row = cursor.execute(sql, (meeting_id, quote)).fetchone()
    return dict(row) if row else None


def snap_quote_to_meeting_turn(
    meeting_id: str,
    imperfect_quote: str,
    min_score: float = 80.0,
    db_path: Optional[Path] = None,
) -> Optional[Tuple[str, int, str, float]]:
    """
    Auto-snapper: searches all speaker turns in the target meeting for the best
    fuzzy partial match and extracts the EXACT verbatim character slice.

    Returns:
        (exact_verbatim_quote, turn_id, speaker, alignment_score) or None
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Fetch all turns for this meeting (typically 200 - 900 turns, fast in-memory)
    rows = cursor.execute(
        "SELECT turn_id, speaker, text FROM turns WHERE meeting_id = ? ORDER BY turn_id",
        (meeting_id,)
    ).fetchall()

    if not rows:
        return None

    cleaned_quote = imperfect_quote.strip()
    quote_lower = cleaned_quote.lower()

    best_score = 0.0
    best_turn_id: Optional[int] = None
    best_speaker: Optional[str] = None
    best_slice = ""

    for r in rows:
        turn_text = r["text"]
        if not turn_text:
            continue

        turn_lower = turn_text.lower()
        alignment = fuzz.partial_ratio_alignment(quote_lower, turn_lower)

        if alignment and alignment.score > best_score:
            best_score = alignment.score
            best_turn_id = r["turn_id"]
            best_speaker = r["speaker"]

            # Expand alignment indices to full word boundaries in source text
            start = alignment.dest_start
            end = alignment.dest_end

            while start > 0 and turn_text[start - 1].isalnum():
                start -= 1
            while end < len(turn_text) and turn_text[end].isalnum():
                end += 1

            best_slice = turn_text[start:end].strip()

    if best_score >= min_score and best_slice:
        # Confirm that snapped quote has sufficient words
        if count_words(best_slice) >= MIN_CITATION_WORDS:
            return best_slice, best_turn_id, best_speaker, best_score

    return None


def verify_citation(
    meeting_id: str,
    quote: str,
    snap: bool = True,
    db_path: Optional[Path] = None,
) -> VerificationResult:
    """
    Verify a single citation against the meeting database.
    If `snap=True`, repairs minor LLM quotation discrepancies back to exact verbatim turns.
    """
    meeting_id = meeting_id.strip()
    raw_quote = quote.strip()
    word_count = count_words(raw_quote)

    if not meeting_id:
        return VerificationResult(
            valid=False,
            meeting_id=meeting_id,
            quote=raw_quote,
            original_quote=raw_quote,
            word_count=word_count,
            reason="Missing meeting_id",
        )

    if word_count < MIN_CITATION_WORDS:
        return VerificationResult(
            valid=False,
            meeting_id=meeting_id,
            quote=raw_quote,
            original_quote=raw_quote,
            word_count=word_count,
            reason=f"Quote has {word_count} words; minimum required is {MIN_CITATION_WORDS}",
        )

    # Fast-path: Check exact substring match
    exact_match = find_exact_quote_in_turn(meeting_id, raw_quote, db_path)
    if exact_match:
        return VerificationResult(
            valid=True,
            meeting_id=meeting_id,
            quote=raw_quote,
            original_quote=raw_quote,
            word_count=word_count,
            turn_id=exact_match["turn_id"],
            speaker=exact_match["speaker"],
            snapped=False,
            similarity_score=100.0,
            reason="Exact verbatim match found in speaker turn",
        )

    # Slow-path: Try auto-snapping if enabled
    if snap:
        snapped_result = snap_quote_to_meeting_turn(meeting_id, raw_quote, min_score=80.0, db_path=db_path)
        if snapped_result:
            exact_quote, turn_id, speaker, score = snapped_result
            return VerificationResult(
                valid=True,
                meeting_id=meeting_id,
                quote=exact_quote,
                original_quote=raw_quote,
                word_count=count_words(exact_quote),
                turn_id=turn_id,
                speaker=speaker,
                snapped=True,
                similarity_score=round(score, 2),
                reason=f"Snapped to verbatim turn text (similarity: {score:.1f}%)",
            )

    return VerificationResult(
        valid=False,
        meeting_id=meeting_id,
        quote=raw_quote,
        original_quote=raw_quote,
        word_count=word_count,
        reason=f"Quote not found verbatim within any single speaker turn in {meeting_id}",
    )


def verify_and_repair_citations(
    citations: List[Dict[str, str]],
    snap: bool = True,
    max_citations: int = MAX_CITATIONS_PER_ANSWER,
    db_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, str]], List[VerificationResult]]:
    """
    Validate and repair a list of candidate citations for an answer.
    Filters out invalid citations and snaps slightly altered quotes to 100% exact text.
    Enforces count bounds (1 to 3 citations).
    """
    repaired_citations: List[Dict[str, str]] = []
    results: List[VerificationResult] = []

    for cit in citations:
        m_id = cit.get("meeting_id", "")
        quote = cit.get("quote", "")

        res = verify_citation(m_id, quote, snap=snap, db_path=db_path)
        results.append(res)

        if res.valid:
            repaired_citations.append({"meeting_id": res.meeting_id, "quote": res.quote})

    # Deduplicate citations while preserving order
    seen = set()
    unique_citations = []
    for c in repaired_citations:
        key = (c["meeting_id"], c["quote"])
        if key not in seen:
            seen.add(key)
            unique_citations.append(c)

    # Cap to max allowed citations
    final_citations = unique_citations[:max_citations]

    return final_citations, results


def validate_answers_file(answers_file_path: Path) -> Dict[str, Any]:
    """
    Audit an entire answers.json submission file for citation compliance.
    Returns comprehensive metrics (total questions, citation validity rate, errors).
    """
    if not answers_file_path.exists():
        raise FileNotFoundError(f"Answers file not found: {answers_file_path}")

    with open(answers_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    answers = data.get("answers", [])
    total_answers = len(answers)
    total_citations = 0
    valid_citations = 0
    errors: List[Dict[str, Any]] = []

    for ans in answers:
        q_id = ans.get("question_id", "UNKNOWN")
        cits = ans.get("citations", [])

        if not (1 <= len(cits) <= MAX_CITATIONS_PER_ANSWER):
            errors.append({
                "question_id": q_id,
                "error": f"Invalid citation count: {len(cits)} (must be 1 to {MAX_CITATIONS_PER_ANSWER})",
            })

        for c in cits:
            total_citations += 1
            res = verify_citation(c.get("meeting_id", ""), c.get("quote", ""), snap=False)
            if res.valid:
                valid_citations += 1
            else:
                errors.append({
                    "question_id": q_id,
                    "meeting_id": c.get("meeting_id"),
                    "quote": c.get("quote"),
                    "reason": res.reason,
                })

    validity_rate = (valid_citations / total_citations * 100) if total_citations > 0 else 0.0

    return {
        "total_answers": total_answers,
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "validity_rate_pct": round(validity_rate, 2),
        "errors_count": len(errors),
        "errors": errors,
    }


if __name__ == "__main__":
    from rich import print as rprint

    rprint("[bold blue]Testing Verbatim Citation Verifier...[/bold blue]")

    # Test 1: Exact Verbatim Quote from TS3007b
    exact_quote = "Uh first you have the energy source."
    res1 = verify_citation("TS3007b", exact_quote, snap=False)
    rprint(f"Test 1 (Exact): Valid={res1.valid}, Turn={res1.turn_id}, Speaker={res1.speaker}")

    # Test 2: Slightly imperfect quote (Auto-snapping enabled)
    # LLM drops 'Uh' and period
    imperfect_quote = "first you have the energy source The energy source would be a battery"
    res2 = verify_citation("TS3007b", imperfect_quote, snap=True)
    rprint(f"Test 2 (Snapped): Valid={res2.valid}, Snapped={res2.snapped}, Score={res2.similarity_score}%")
    rprint(f"  Original: [red]{imperfect_quote}[/red]")
    rprint(f"  Snapped:  [green]{res2.quote}[/green]")

    # Test 3: Less than 6 words (Must fail)
    short_quote = "first you have"
    res3 = verify_citation("TS3007b", short_quote, snap=True)
    rprint(f"Test 3 (Too Short): Valid={res3.valid}, Reason='{res3.reason}'")
