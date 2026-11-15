# Temporal Memory and State Ledger for Catalist Meeting Memory

# Core Capabilities:
# 1. Chronological Timeline Builder: Retrieves and organizes evidence turns strictly
#    across meeting sessions (A -> B -> C -> D) without context stuffing (scales to 10k+ meetings)
# 2. Speaker Role Resolver: Extracts speaker identities and roles (Project Manager,
#    User Interface Designer, Industrial Designer, Marketing) from kick-off meetings (Instruction #4)
# 3. Pre-Final Brief Context Extractor: Assembles chronological context up to Meeting C
#    (excluding Meeting D) for pre-final briefs like Team TS3012


from __future__ import annotations

import re
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

from src.ingest import (
    search_turns,
    get_turn_context,
    get_team_turns,
    get_db_connection,
)

# Known AMI Role patterns
ROLE_PATTERNS = {
    "Project Manager": re.compile(r"project manager", re.IGNORECASE),
    "User Interface Designer": re.compile(r"(user interface|interface designer|ui designer)", re.IGNORECASE),
    "Industrial Designer": re.compile(r"(industrial designer|id designer)", re.IGNORECASE),
    "Marketing": re.compile(r"(marketing|marketing expert)", re.IGNORECASE),
}


@dataclass
class TimelineTurn:
    meeting_id: str
    session: str
    session_seq: int
    turn_id: int
    timestamp: str
    speaker: str
    speaker_role: str
    text: str
    is_focal: bool = True  # True if directly matched query, False if surrounding context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meeting_id": self.meeting_id,
            "session": self.session,
            "session_seq": self.session_seq,
            "turn_id": self.turn_id,
            "timestamp": self.timestamp,
            "speaker": self.speaker,
            "speaker_role": self.speaker_role,
            "text": self.text,
            "is_focal": self.is_focal,
        }


@dataclass
class TeamTemporalTimeline:
    team: str
    query: str
    roles: Dict[str, str]
    turns_by_session: Dict[str, List[TimelineTurn]] = field(default_factory=dict)
    all_turns: List[TimelineTurn] = field(default_factory=list)

    def to_prompt_text(self, max_chars: int = 12000) -> str:
        """Format the chronological timeline into structured context for LLM prompt."""
        role_lines = [f"- {spk}: {role}" for spk, role in self.roles.items() if role]
        roles_block = "\n".join(role_lines) if role_lines else "- Roles not explicitly introduced."

        session_names = {
            "a": "Meeting 1 (a) - Kick-off & Introductions",
            "b": "Meeting 2 (b) - Functional Design & Initial Brainstorming",
            "c": "Meeting 3 (c) - Conceptual Design & Component Debates",
            "d": "Meeting 4 (d) - Detailed Design, Costing & Final Decisions",
        }

        lines = [
            f"=== CHRONOLOGICAL MEETING TIMELINE FOR TEAM {self.team} ===",
            f"Topic/Keywords: {self.query}",
            "",
            "--- Speaker Identifications & Roles ---",
            roles_block,
            "",
            "--- Sequential Evidence Across Meetings ---",
        ]

        total_length = 0
        for session in ["a", "b", "c", "d"]:
            turns = self.turns_by_session.get(session, [])
            if not turns:
                continue

            session_title = session_names.get(session, f"Meeting ({session})")
            lines.append(f"\n### [{session_title} - Meeting ID: {turns[0].meeting_id}]")

            for t in turns:
                role_tag = f" ({t.speaker_role})" if t.speaker_role else ""
                entry = f"  [{t.meeting_id} | Turn {t.turn_id} | {t.timestamp}] {t.speaker}{role_tag}: \"{t.text}\""
                lines.append(entry)
                total_length += len(entry)

                if total_length > max_chars:
                    lines.append("  ... [Remaining turns truncated for length limit] ...")
                    break

        return "\n".join(lines)


# ==============================================
# Speaker & Role Resolution

# AMI Core Roles
AMI_ROLES = ["Project Manager", "Industrial Designer", "User Interface Designer", "Marketing Expert"]

FIRST_PERSON_INTRO_PATTERNS = [
    (re.compile(r"\b(i am|i'm|i'll be( your)?|my name('s| is))\b.*?\bproject manager\b", re.I), "Project Manager"),
    (re.compile(r"\b(i am|i'm|i'll be( your)?|my name('s| is)|presented by me, the)\b.*?\bindustrial designer\b", re.I), "Industrial Designer"),
    (re.compile(r"\b(i am|i'm|i'll be( your)?|my name('s| is))\b.*?\b(user )?interface designer\b", re.I), "User Interface Designer"),
    (re.compile(r"\b(i am|i'm|i'll be( your)?|my name('s| is))\b.*?\bmarketing (expert)?\b", re.I), "Marketing Expert"),
]

TOPIC_PRESENTATION_PATTERNS = [
    (re.compile(r"\b(working design|components?|technical functions?|hardware|circuit|infrared bulb|chip)\b", re.I), "Industrial Designer", 3),
    (re.compile(r"\b(user interface|interface design|g_u_i_|button layout|display screen|menu navigation)\b", re.I), "User Interface Designer", 3),
    (re.compile(r"\b(market research|target group|user requirements|market trends?|trend-watching|demographics)\b", re.I), "Marketing Expert", 3),
    (re.compile(r"\b(agenda for today|project plan|welcome (at|to) the kick-?off|wrap up the meeting)\b", re.I), "Project Manager", 5),
]


def get_team_roles(team: str, db_path: Optional[Any] = None) -> Dict[str, str]:
    """
    Extract speaker roles (Project Manager, User Interface Designer, Industrial Designer, Marketing Expert)
    by scanning kick-off introductions (Session A) and presentation openings across Sessions A, B, and C.
    Employs deductive assignment so all 4 speakers receive distinct functional roles.
    """
    team_clean = team.upper().strip()
    speakers = ["Speaker A", "Speaker B", "Speaker C", "Speaker D"]
    scores = {spk: {r: 0 for r in AMI_ROLES} for spk in speakers}

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    rows = cursor.execute("""
        SELECT session, turn_id, speaker, text
        FROM turns
        WHERE team = ? AND session IN ('a', 'b', 'c')
        ORDER BY session_seq ASC, turn_id ASC
    """, (team_clean,)).fetchall()

    for r in rows:
        spk = r["speaker"]
        if spk not in scores:
            continue
        text = r["text"]

        # High-weight first-person introductions in early kick-off turns
        if r["session"] == 'a' and r["turn_id"] <= 60:
            for pat, role in FIRST_PERSON_INTRO_PATTERNS:
                if pat.search(text):
                    scores[spk][role] += 100

        # Topic and presentation opening cues
        for pat, role, weight in TOPIC_PRESENTATION_PATTERNS:
            if pat.search(text):
                w = weight * 2 if r["turn_id"] <= 120 else weight
                scores[spk][role] += w

    # Greedy bipartite matching
    assigned_roles: Dict[str, str] = {spk: "" for spk in speakers}
    rem_speakers = set(speakers)
    rem_roles = set(AMI_ROLES)

    candidates = []
    for spk in speakers:
        for role in AMI_ROLES:
            candidates.append((scores[spk][role], spk, role))
    candidates.sort(reverse=True)

    for score, spk, role in candidates:
        if score > 0 and spk in rem_speakers and role in rem_roles:
            assigned_roles[spk] = role
            rem_speakers.remove(spk)
            rem_roles.remove(role)

    # Deductive completion if 3 assigned and 1 remains
    if len(rem_speakers) == 1 and len(rem_roles) == 1:
        s = rem_speakers.pop()
        assigned_roles[s] = rem_roles.pop()

    return assigned_roles


# =====================================================================
# Temporal Timeline Construction


def build_temporal_timeline(
    team: str,
    query: str,
    max_session: Optional[str] = None,
    max_turns_per_session: int = 5,
    window: int = 1,
    db_path: Optional[Any] = None,
) -> TeamTemporalTimeline:
    """
    Construct a deduplicated, chronological timeline of turns matching the query
    across meetings A -> B -> C -> D with dialogue context windows.
    """
    team_clean = team.upper().strip()
    roles = get_team_roles(team_clean, db_path=db_path)

    # 1. Search FTS5 for candidate matches
    hits = search_turns(query, team=team_clean, limit=40, db_path=db_path)

    # Filter by max_session if specified (e.g. 'c' for pre-final briefs)
    session_order = {"a": 1, "b": 2, "c": 3, "d": 4}
    max_seq = session_order.get(max_session.lower(), 4) if max_session else 4
    valid_hits = [h for h in hits if h["session_seq"] <= max_seq]

    # 2. Group focal hits by session and pick top N per session
    focal_by_session: Dict[str, List[Dict[str, Any]]] = {}
    for h in valid_hits:
        s = h["session"]
        focal_by_session.setdefault(s, []).append(h)

    # 3. For top focal hits, expand surrounding context
    turn_keys_seen: Set[str] = set()
    collected_turns: List[TimelineTurn] = []

    for s in ["a", "b", "c", "d"]:
        if session_order[s] > max_seq:
            continue

        s_hits = focal_by_session.get(s, [])[:max_turns_per_session]
        for focal in s_hits:
            m_id = focal["meeting_id"]
            t_id = focal["turn_id"]

            # Expand window around focal turn
            context_turns = get_turn_context(m_id, t_id, window=window, db_path=db_path)
            for c in context_turns:
                key = f"{c['meeting_id']}:{c['turn_id']}"
                if key in turn_keys_seen:
                    continue
                turn_keys_seen.add(key)

                spk = c["speaker"]
                spk_role = roles.get(spk, "")
                is_focal = (c["turn_id"] == t_id)

                # Determine timestamp from context
                collected_turns.append(TimelineTurn(
                    meeting_id=c["meeting_id"],
                    session=s,
                    session_seq=session_order[s],
                    turn_id=c["turn_id"],
                    timestamp=c["timestamp"],
                    speaker=spk,
                    speaker_role=spk_role,
                    text=c["text"],
                    is_focal=is_focal,
                ))

    # 4. Sort turns chronologically
    collected_turns.sort(key=lambda t: (t.session_seq, t.turn_id))

    # 5. Group by session
    turns_by_session: Dict[str, List[TimelineTurn]] = {}
    for t in collected_turns:
        turns_by_session.setdefault(t.session, []).append(t)

    return TeamTemporalTimeline(
        team=team_clean,
        query=query,
        roles=roles,
        turns_by_session=turns_by_session,
        all_turns=collected_turns,
    )


# =====================================================================
# Pre-Final Brief Context (Team TS3012)

def get_pre_final_brief_context(
    team: str = "TS3012",
    db_path: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Extract comprehensive pre-final briefing context for a team across meetings A, B, and C.
    Explicitly excludes meeting D to ensure zero look-ahead data leakage.
    """
    team_clean = team.upper().strip()
    roles = get_team_roles(team_clean, db_path=db_path)

    # Keywords covering design decisions, cost, components, and actions
    topics = [
        ("components", "battery kinetic solar power speech screen button shape material"),
        ("costs", "cost budget price euro expense limit expensive"),
        ("actions", "action agree investigate evaluate decide will look find"),
    ]

    all_timeline_turns: List[TimelineTurn] = []
    seen_keys: Set[str] = set()

    for topic_name, q in topics:
        timeline = build_temporal_timeline(
            team=team_clean,
            query=q,
            max_session="c",  # Strictly exclude meeting D
            max_turns_per_session=6,
            window=1,
            db_path=db_path,
        )
        for t in timeline.all_turns:
            k = f"{t.meeting_id}:{t.turn_id}"
            if k not in seen_keys:
                seen_keys.add(k)
                all_timeline_turns.append(t)

    all_timeline_turns.sort(key=lambda t: (t.session_seq, t.turn_id))

    return {
        "team": team_clean,
        "roles": roles,
        "total_turns": len(all_timeline_turns),
        "turns": all_timeline_turns,
    }


if __name__ == "__main__":
    from rich import print as rprint

    rprint("[bold blue]Testing Temporal Memory Ledger for Team TS3007...[/bold blue]")
    roles = get_team_roles("TS3007")
    rprint("[bold cyan]Detected Roles for TS3007:[/bold cyan]", roles)

    timeline = build_temporal_timeline("TS3007", "battery kinetic energy")
    rprint(f"[bold green]Timeline Built:[/bold green] {len(timeline.all_turns)} turns retrieved across {len(timeline.turns_by_session)} sessions.")
    
    prompt_text = timeline.to_prompt_text(max_chars=2000)
    rprint("\n[bold yellow]Sample Prompt Representation (First 1,500 chars):[/bold yellow]")
    rprint(prompt_text[:1500])
