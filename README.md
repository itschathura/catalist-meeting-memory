# Catalist Meeting Memory

[![Tests](https://img.shields.io/badge/pytest-24%20passed-brightgreen.svg)]()
[![Citations](https://img.shields.io/badge/citation%20validity-100%25-blue.svg)]()
[![Model](https://img.shields.io/badge/Gemini-3.5%20Flash%20Lite-orange.svg)]()
[![Scale](https://img.shields.io/badge/Corpus-138%20meetings%20%7C%2048k%20turns-purple.svg)]()

> **Scale-ready agentic memory system over 138 AMI meeting transcripts (~676,000 words / ~880,000 tokens) designed for 10,000+ meetings without context stuffing.**  
> Built for the **Catalist Agentic AI Intern Build Test**.

---

## 🌟 Quickstart & Running the Project

### 1. Installation
Clone the repository and install the lightweight dependencies:
```bash
pip install -r requirements.txt
```

Set your Gemini API Key in `.env`:
```env
GEMINI_API_KEY=your_actual_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

### 2. Ingest & Index Transcripts (0.87 seconds)
Parses all 138 meeting transcripts into SQLite relational tables and creates an FTS5 BM25 search index:
```bash
python -m src.ingest
```

### 3. Run Benchmark (Official 25 Questions)
Executes all 25 official benchmark questions, verifies citations down to verbatim character borders, embeds the TS3012 brief, and generates `answers.json`:
```bash
python run.py
```

### 4. Interactive Live Demo CLI (For Interview)
Test live questions during the interview with Abdul:
```bash
# Targeted CLI Query
python live_demo.py --team TS3007 --q "Team TS3007's energy source changed between meetings. What did they first settle on, what did they end with, and who proposed the change?"

# Interactive Mode
python live_demo.py
```

### 5. Automated Evaluation Suite
Run evaluation on self-generated benchmark questions or official questions:
```bash
# Evaluate 10 self-generated ground truth questions
python -m eval.run_eval --my-questions

# Run full project test suite (24 tests)
pytest tests/ -v
```

---

## 📂 Deliverables & Repository Structure

* **[answers.json](answers.json)**: Official submission JSON containing all 25 benchmark answers with 100% machine-verified verbatim citations and the embedded TS3012 brief.
* **[brief_ts3012.md](brief_ts3012.md)**: Dedicated Pre-Final Meeting Brief for Team TS3012 (Decisions, Changes, Action Items, Disagreements) with zero look-ahead data leakage.
* **[eval/my_questions.json](eval/my_questions.json)**: 10 self-generated ground-truth benchmark questions across remaining teams.
* **[failures.md](failures.md)**: Engineering post-mortem detailing 3 authentic failure modes encountered and resolved.
* **[WRITEUP.md](WRITEUP.md)**: Comprehensive architectural report, tool-fit judgment (code vs. model), and empirical scaling analysis for 1,000 to 10,000 meetings.

---

## 📊 Evaluation & Empirical Results

| Metric | Official 25 Questions (`answers.json`) | 10 Self-Generated Questions (`my_questions.json`) |
| :--- | :---: | :---: |
| **Questions Answered** | **25 / 25 (100%)** | **10 / 10 (100%)** |
| **Total Citations** | **45** | **15** |
| **Citation Validity Rate** | **100.0% Exact Verbatim** | **100.0% Exact Verbatim** |
| **Corpus Ingestion Time** | **0.871s** (48k turns) | **0.871s** (48k turns) |
| **Average Query Latency** | **~2.8s** | **2.28s** |
| **Total Token Cost** | **<$0.015 USD** | **$0.0041 USD** |
| **Automated Unit Tests** | **24 / 24 PASSED (0.77s)** | **24 / 24 PASSED (0.77s)** |
