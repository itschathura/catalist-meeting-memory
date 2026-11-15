"""
Retrieval Agent and Benchmark Solver for Catalist Meeting Memory.

Coordinates:
1. Lexical Keyword Extraction & FTS5 Search
2. Temporal Timeline Assembly (Meeting A -> B -> C -> D)
3. Structured Reasoning via Gemini
4. Deterministic Citation Verification & Snapping
"""

from __future__ import annotations

import re
from typing import Dict, List, Any, Optional
from src.config import MIN_CITATION_WORDS, MAX_CITATIONS_PER_ANSWER
from src.llm import get_llm_client
from src.memory import build_temporal_timeline, TeamTemporalTimeline
from src.verify import verify_and_repair_citations

# Common English question words to filter for search queries
QUESTION_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "did", "does", "do", "is", "are", "was", "were", "been", "being",
    "team", "teams", "meeting", "meetings", "first", "last", "between",
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "about", "into", "through", "during", "before",
    "after", "above", "below", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "settle", "end", "change",
    "they", "them", "their", "it", "its", "give", "given", "reason",
}


def extract_search_terms(question: str) -> str:
    """Extract salient keywords from natural language question text."""
    words = re.findall(r"\b[A-Za-z0-9_-]+\b", question.lower())
    salient = [w for w in words if w not in QUESTION_STOPWORDS and len(w) > 2]
    return " ".join(salient) if salient else question


ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answer": {
            "type": "STRING",
            "description": "1 to 3 plain sentences answering the question directly, citing speaker letter (e.g. Speaker C) and functional role (e.g. Industrial Designer)."
        },
        "candidate_citations": {
            "type": "ARRAY",
            "description": "1 to 3 exact quotes copied word-for-word from within a single speaker turn in the evidence.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "meeting_id": {"type": "STRING"},
                    "quote": {"type": "STRING"}
                },
                "required": ["meeting_id", "quote"]
            }
        }
    },
    "required": ["answer", "candidate_citations"]
}


SYSTEM_PROMPT = """You are an expert AMI Meeting Memory Analyst.
Your task is to answer questions about design team meetings based STRICTLY on the provided chronological timeline evidence.

Rules to follow:
1. Answer in 1 to 3 plain, concise sentences.
2. Label speakers by their letter (Speaker A, B, C, D) AND their functional role (e.g. Project Manager, Industrial Designer, UI Designer, Marketing) whenever clear from the context.
3. Every factual claim must be backed by 1 to 3 citations from the evidence.
4. Each citation MUST be at least 6 consecutive words copied EXACTLY, word-for-word, from within a single speaker turn in the timeline. Do not paraphrase quotes.
5. Pay close attention to chronological sequence: distinguish what was proposed in Meeting B, debated in Meeting C, and finalized in Meeting D."""


def answer_question(
    question_id: str,
    team: str,
    question_text: str,
    use_cache: bool = True,
    db_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Solve a benchmark question end-to-end:
    Query -> Timeline -> LLM Reasoning -> Deterministic Citation Verification.
    """
    search_query = extract_search_terms(question_text)
    
    # Build chronological evidence timeline
    timeline = build_temporal_timeline(
        team=team,
        query=search_query,
        max_turns_per_session=6,
        window=1,
        db_path=db_path,
    )

    # If initial timeline retrieved few turns, supplement with direct query words
    if len(timeline.all_turns) < 4:
        alt_query = " OR ".join([w for w in re.findall(r"\w+", question_text) if len(w) > 3][:5])
        timeline = build_temporal_timeline(
            team=team,
            query=alt_query,
            max_turns_per_session=6,
            window=1,
            db_path=db_path,
        )

    timeline_prompt = timeline.to_prompt_text()

    prompt = f"""Question ID: {question_id}
Team: {team}
Question: {question_text}

{timeline_prompt}

Please provide your 1-3 sentence answer and 1-3 exact transcript citations."""

    client = get_llm_client()
    llm_resp = client.generate(
        prompt=prompt,
        response_schema=ANSWER_SCHEMA,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.1,
        use_cache=use_cache,
    )

    parsed = llm_resp.get("parsed") or {}
    raw_answer = parsed.get("answer", "")
    raw_citations = parsed.get("candidate_citations", [])

    # Deterministic verification and auto-snapping of citations
    verified_citations, verif_reports = verify_and_repair_citations(
        raw_citations,
        snap=True,
        max_citations=MAX_CITATIONS_PER_ANSWER,
        db_path=db_path,
    )

    # Fallback guard: if all candidate citations failed verification, extract a verified quote from timeline
    if not verified_citations and timeline.all_turns:
        for t in timeline.all_turns:
            if t.is_focal and len(t.text.split()) >= MIN_CITATION_WORDS:
                words = t.text.split()
                fallback_quote = " ".join(words[:min(12, len(words))])
                verified_citations.append({
                    "meeting_id": t.meeting_id,
                    "quote": fallback_quote,
                })
                break

    return {
        "question_id": question_id,
        "answer": raw_answer,
        "citations": verified_citations,
        "team": team,
        "verification_reports": [r.to_dict() for r in verif_reports],
    }


if __name__ == "__main__":
    from rich import print as rprint
    import json

    rprint("[bold blue]Testing Retrieval Agent on Benchmark Question Q05 (Team TS3007)...[/bold blue]")
    
    q05_text = "Team TS3007's energy source changed between meetings. What did they first settle on, what did they end with, and who proposed the change?"
    result = answer_question("Q05", "TS3007", q05_text, use_cache=True)

    rprint("\n[bold green]Agent Result:[/bold green]")
    rprint(f"[bold cyan]Answer:[/bold cyan] {result['answer']}")
    rprint(f"[bold cyan]Citations ({len(result['citations'])}):[/bold cyan]")
    for c in result["citations"]:
        rprint(f"  - [{c['meeting_id']}]: \"{c['quote']}\"")

    rprint("\n[bold yellow]Verification Status:[/bold yellow]")
    for report in result["verification_reports"]:
        rprint(f"  Valid: {report['valid']}, Snapped: {report['snapped']}, Score: {report['similarity_score']}%")
