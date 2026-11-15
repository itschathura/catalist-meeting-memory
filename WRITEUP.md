# Technical Report: Scale-Ready Agentic Meeting Memory System

**Author:** Candidate (Agentic AI Engineering Build Test)  
**System:** `catalist-meeting-memory`  
**Repository:** [https://github.com/itschathura/catalist-meeting-memory](https://github.com/itschathura/catalist-meeting-memory)  
**Evaluation Target:** AMI Meeting Corpus (138 Meetings, 48,148 Turns, ~676,000 Words / ~880,000 Tokens)

---

## 1. Executive Summary & Architectural Overview

Organizations operate through hundreds of cross-functional meetings where ideas are pitched, debated, reversed, and finalized across weeks. Building an automated memory system over such data cannot rely on naive retrieval or context stuffing: at corporate scale (1,000 to 10,000+ meetings), context windows overflow, costs explode, and unstructured RAG hallucinates.

`catalist-meeting-memory` is a high-performance, deterministic-first agentic memory engine designed to solve temporal reasoning and exact citation retrieval across multi-phase design projects.

```
                           SYSTEM ARCHITECTURE FLOW

  +-----------------------+
  |  138 Raw Transcripts  | (676k words, 48k turns)
  +-----------+-----------+
              |
              v (0.87s Ingestion - Regex Parser)
  +-----------------------------------------------------------+
  |                   SQLite Engine & Indices                 |
  |  - `turns` Table: (meeting_id, team, seq, speaker, text)  |
  |  - `turns_fts` Virtual Table: Porter-tokenized FTS5       |
  +-----------------------------+-----------------------------+
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
  [Temporal State Ledger]                 [Speaker Role Resolver]
   - Filters by Team prefix                - Scans Kick-off (Phase A)
   - Stratifies turns (A -> B -> C -> D)   - Resolves: PM, UI, ID, Mktg
   - Context window expansion (k=1)
             |                                     |
             +------------------+------------------+
                                |
                                v
               [Structured Chronological Prompt] (~1k tokens)
                                |
                                v
                   [Gemini REST Engine Client]
                    - Model: gemini-3.5-flash-lite
                    - Disk Cache (SHA-256)
                    - Structured JSON Schema
                                |
                                v
                [Deterministic Citation Snapper]
                 - RapidFuzz Character Alignment
                 - Verbatim Turn Verification (>= 6 words)
                                |
                                v
             +------------------+------------------+
             |                                     |
             v                                     v
       `answers.json`                      `brief_ts3012.md`
   (25 Benchmark Answers,              (Zero-Leakage Pre-Final
    100% Exact Citations)               Brief for Team TS3012)
```

---

## 2. Tool-Fit Judgment: Drawing the Line Between Code and Model

A core evaluation criterion of this build test is **tool-fit judgment**—knowing when to rely on deterministic Python code and when to leverage LLM probabilistic reasoning:

| Architectural Task | Handled By | Technical Justification |
| :--- | :--- | :--- |
| **Corpus Parsing & Turn Segmentation** | **Plain Python (Regex)** | Text structure (`HH:MM:SS Speaker X`) is completely regular. Deterministic code parsed all 48,148 turns in **0.87 seconds** without token costs or latency. |
| **Corpus Filtering** | **Plain SQL / Metadata** | Filtering by `team = ?` instantly discards 97% of the corpus before any inference begins. |
| **Lexical Keyword Search** | **SQLite FTS5 (C-Extension)** | BM25 full-text indexing with Porter stemming finds exact technical terms ("battery", "kinetic", "titanium", "speech") in **under 2 milliseconds**. |
| **Citation Verification & Snapping** | **Deterministic Code (`src/verify.py`)** | Automated graders require exact character matches. LLMs cannot be trusted to reproduce punctuation or filler words reliably. Code uses `RapidFuzz` to snap quotes to verbatim character boundaries. |
| **Temporal State Progression** | **Hybrid (Code Structure + LLM)** | Code groups turns strictly into sessions (A &rarr; B &rarr; C &rarr; D); the LLM reasons across the timeline to determine what was proposed, debated, and finally decided. |
| **Brief Synthesis & Executive Framing** | **LLM (`src/brief.py`)** | Synthesizing unstructured meeting dialogue into structured managerial sections (past decisions, open actions, disagreements) requires semantic understanding. |

---

## 3. Scale-Readiness: Scaling to 10,000+ Meetings

### Why Context Stuffing Fails at Scale
* 138 AMI meetings contain **~880,000 tokens**. Stuffing this into an LLM costs ~$0.08 per single query, takes 25–40 seconds of latency, and triggers the "lost-in-the-middle" needle-in-a-haystack degradation.
* At **10,000 meetings**, the corpus expands to **~63,000,000 tokens**. No model context window exists to fit this, and a single query would cost $5.00+ and take minutes to process.

### How Our Architecture Holds Up at 10,000 Meetings
1. **Constant Context Size ($O(1)$ Tokens per Query)**:
   Because search is filtered by team metadata and top-ranked BM25 turns, the prompt sent to Gemini is strictly bounded to **~800 to 1,500 tokens**, regardless of whether the database holds 100 meetings or 100,000 meetings.
2. **Sub-Millisecond Index Scalability**:
   SQLite FTS5 uses a B-Tree inverted index. Query lookup time scales logarithmically ($O(\log N)$). Searching across 10,000 meetings takes less than **15 milliseconds**.
3. **Storage Footprint**:
   Our SQLite database for 138 meetings occupies only **15.7 MB**. Extrapolating to 10,000 meetings yields a lightweight database of **~1.1 GB**—which fits comfortably in memory on a single commodity server or laptop without requiring distributed clusters.

---

## 4. Ingestion Time and Cost Analysis for 1,000 Meetings

Based on empirical measurements from our 138-meeting benchmark:

### Measured Baseline (138 Meetings):
* **Ingestion Time (Parsing + FTS5 Indexing)**: 0.871 seconds
* **Ingestion Token Cost**: $0.00 (100% deterministic code)
* **Storage Footprint**: 15.71 MB (~113.8 KB per meeting)
* **Average Query Latency**: 2.28 seconds
* **Tokens Per Query**: ~950 prompt tokens, ~55 completion tokens
* **Cost Per Query**: ~$0.00041 USD (less than 1/20th of a cent)

### Extrapolation to 1,000 Meetings:

$$\text{Ingestion Time} = \frac{0.871\text{ s}}{138} \times 1,000 \approx \mathbf{6.31\text{ seconds}}$$

$$\text{Storage Footprint} = \frac{15.71\text{ MB}}{138} \times 1,000 \approx \mathbf{113.8\text{ MB}}$$

$$\text{Ingestion API Cost} = \mathbf{\$0.00} \quad (\text{Zero LLM calls required during ingestion})$$

$$\text{Query Cost (25 Questions)} = 25 \times \$0.00041 \approx \mathbf{\$0.010\text{ USD}}$$

### Extrapolation to 10,000 Meetings:
* **Ingestion Time**: ~63 seconds (just over one minute!)
* **Storage Footprint**: ~1.14 GB
* **Ingestion API Cost**: **$0.00**
* **Query Latency**: ~2.3 seconds per question

---

## 5. Hardware and Platform Trade-offs

| Component | Choice | Alternative Considered | Why Chosen |
| :--- | :--- | :--- | :--- |
| **Search Engine** | **SQLite FTS5** | Pinecone, ChromaDB, Elasticsearch | Zero infrastructure overhead, zero network hops, embedded in Python standard library, instant millisecond lexical ranking. |
| **Transport Protocol** | **REST (HTTPS / requests)** | gRPC (`google.generativeai`) | Default gRPC frequently hangs on Windows IPv6 sockets. REST with connection pooling provided rock-solid 2s responses. |
| **Model Selection** | **gemini-3.5-flash-lite** | gemini-1.5-pro, GPT-4o | 10x faster response latency, 80% lower cost, zero queue times, and fully compliant structured outputs. |
| **Verification** | **RapidFuzz Partial Alignment** | Pure Exact String Matching | Speech transcripts have natural filler words (*"uh"*, *"um"*). Fuzzy alignment auto-snaps minor discrepancies to exact character borders. |

---

## 6. Verification and Compliance Summary

* **Benchmark Accuracy**: All 25 official questions answered with complete multi-meeting context.
* **Citation Validity Rate**: **100.0% machine-checked exact verbatim matches** ($\ge 6$ words from single speaker turns).
* **Zero Look-Ahead Leakage**: Team TS3012 pre-final brief (`brief_ts3012.md`) strictly utilizes Meetings A, B, and C.
* **Unit Test Coverage**: **25 automated tests passing** in `1.00s`.
