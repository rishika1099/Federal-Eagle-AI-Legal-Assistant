# 🦅 Federal Eagle

### An AI-powered U.S. Federal Law Analysis Assistant

**Federal Eagle** is a multi-agent legal analysis assistant that helps users understand the potential U.S. federal law implications of real-world scenarios. It combines structured legal reasoning, semantic statute retrieval, conservative precedent discovery, and practical document drafting, all in one Streamlit interface.

> ⚠️ **Disclaimer**: Federal Eagle is for educational and internal analysis purposes only and does NOT provide legal advice.

---

## ✨ What Federal Eagle Does

Given a plain-English description of a situation, Federal Eagle can:

* 🔍 Summarize the legal issue and classify the case type
* ⚖️ Identify why the matter may fall under U.S. federal jurisdiction
* 📚 Retrieve relevant U.S. Code (USC) sections via hybrid semantic + lexical search
* 🧩 Break statutes into elements and assess whether they appear met, unmet, or unclear
* 🏛️ Surface high-confidence federal judicial precedents (Supreme Court and Circuit Courts)
* 📄 Generate a practical draft document (memo, issue outline, or internal summary)
* ⬇️ Export drafts as Word or text files

All results are structured, explainable, and designed to support human decision-making.

---

## 🏛️ Architecture

### High-level data flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                User scenario                                 │
│                          (free-text fact pattern)                            │
└──────────────────────────────────────────┬───────────────────────────────────┘
                                           │
                              ┌────────────▼────────────┐
                              │  Streamlit UI (app.py)  │
                              └────────────┬────────────┘
                                           │ crew.kickoff(inputs=...)
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       CrewAI sequential pipeline (crew.py)                   │
│                                                                              │
│  ┌─────────────────────────┐                                                 │
│  │  Case Intake Agent      │   gpt-4o-mini, temperature 0, no tools          │
│  │  agents/case_intake_*   │   Input:  user_input string                     │
│  │                         │   Output: structured intake JSON (case_type,    │
│  └────────────┬────────────┘            legal_domain, hooks, queries, ...)   │
│               │                                                              │
│               ▼                                                              │
│  ┌─────────────────────────┐         ┌────────────────────────────────────┐  │
│  │  USC Retrieval Agent    │ ◄─────► │  USC Search Tool                   │  │
│  │  agents/usc_section_*   │         │  tools/usc_sections_search_tool.py │  │
│  │                         │         │  Hybrid retrieval:                 │  │
│  │  Picks 2-4 best queries │         │   (1) direct citation lookup       │  │
│  │  from intake, calls the │         │       (metadata-filtered get)      │  │
│  │  tool per query, merges │         │   (2) MiniLM semantic search       │  │
│  │  + dedupes top 3-5      │         │       (all-MiniLM-L6-v2)           │  │
│  │  statutes               │         │   (3) lexical token-overlap        │  │
│  │                         │         │       fallback                     │  │
│  └────────────┬────────────┘         └────────────────────────────────────┘  │
│               │                                              │               │
│               │                                              ▼               │
│               │                                ┌─────────────────────┐       │
│               │                                │  ChromaDB (local)   │       │
│               │                                │  collection:        │       │
│               │                                │  usc_complete       │       │
│               │                                │  ~4.5k sections     │       │
│               │                                └─────────────────────┘       │
│               ▼                                                              │
│  ┌─────────────────────────┐         ┌────────────────────────────────────┐  │
│  │  Precedent Agent        │ ◄─────► │  Legal Precedent Search Tool       │  │
│  │  agents/legal_*         │         │  tools/legal_precedent_search_*    │  │
│  │                         │         │  Wraps Tavily, restricts to        │  │
│  │  Builds 2-3 queries,    │         │  whitelisted legal-source domains  │  │
│  │  filters out non-opinion│         │  (cornell, justia, courtlistener,  │  │
│  │  pages, returns up to 3 │         │  supremecourt.gov, etc.)           │  │
│  └────────────┬────────────┘         └────────────────────────────────────┘  │
│               │                                                              │
│               ▼                                                              │
│  ┌─────────────────────────┐                                                 │
│  │  Legal Drafter Agent    │   Receives ALL three upstream JSONs as context. │
│  │  agents/legal_drafter_* │   Emits single UI-ready JSON:                   │
│  │                         │     summary, statutes, elements_analysis,       │
│  │                         │     precedents, next_steps, clarifying_questions│
│  │                         │     draft_document, disclaimer                  │
│  └────────────┬────────────┘                                                 │
└───────────────┼──────────────────────────────────────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Excerpt Repair (post-processor)                         │
   │  tools/usc_sections_search_tool.py::repair_drafter_*     │
   │  Deterministically replaces drafter statute excerpts     │
   │  with verbatim contiguous substrings of upstream USC     │
   │  text. Catches paraphrase-as-quote hallucinations.       │
   └────────────┬─────────────────────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Streamlit tabs                                          │
   │    Summary    | Statutes (USC DB)  | Elements            │
   │    Precedents | Draft + Export (Word/TXT)                │
   └──────────────────────────────────────────────────────────┘
```

### Offline data preparation (run once before first use)

```
  USC bulk XML downloads          usc_parser.py                  usc_complete.json
  (uscode.house.gov)        ───►  parses XML files,        ───►  flat JSON of
  one .xml file per title         extracts sections                ~4.5k sections
                                                                          │
                                                                          ▼
                                                              usc_vectordb_builder.py
                                                                          │
                                                                          ▼
                                                                    chroma_db/
                                                              persistent HNSW
                                                              index over MiniLM
                                                              sentence-transformer
                                                              embeddings
```

---

## 🤖 Per-agent detail

| Agent | LLM | Tools | Job |
|---|---|---|---|
| Case Intake | gpt-4o-mini, temp 0 | none | Extract facts, classify case_type, identify federal hooks, propose 4-6 search queries (statute-anchor + plain English + fact-specific). |
| USC Retrieval | gpt-4o-mini, temp 0 | USC Search Tool | Run 2-4 best queries against ChromaDB, merge results, return 3-5 deduped statutes with excerpts. |
| Precedent | gpt-4o-mini, temp 0 | Tavily Legal Search | Build queries, filter to opinion pages on trusted sources, return up to 3 federal precedents with conservative holdings. |
| Drafter | gpt-4o-mini, temp 0 | none | Synthesize final structured JSON; preserve upstream citations and precedents verbatim. |

Determinism: every agent runs at temperature 0. Variance between runs comes from Tavily search results and gpt-4o-mini's small temperature-0 sampling residue.

---

## 📦 Tech Stack

* **Python 3.11**
* **Streamlit** for the interactive UI
* **CrewAI** for multi-agent orchestration
* **OpenAI gpt-4o-mini** for all four agents (roughly $0.002 per end-to-end case)
* **ChromaDB** for persistent semantic statute retrieval
* **sentence-transformers (all-MiniLM-L6-v2)** for embeddings
* **Tavily API** for controlled legal precedent search
* **python-docx** for Word document export

---

## 🛠️ Setup

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/rishika1099/Federal-Eagle-AI-Legal-Assistant.git
cd Federal-Eagle-AI-Legal-Assistant
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Add API keys

Create a `.env` file in the repo root:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
PERSIST_DIRECTORY_PATH=./chroma_db
USC_COLLECTION_NAME=usc_complete
USC_SEARCH_TOP_K=8
TAVILY_MAX_RESULTS=8
TAVILY_SEARCH_DEPTH=advanced
```

### 3. Build the USC vector database (one-time, takes 5-15 minutes)

```bash
# Download USC XML files into the usc/ folder from
# https://uscode.house.gov/download/download.shtml
# Minimum useful set: titles 18, 21, 26, 31. Full set is 50+ titles.

python usc_parser.py --input-dir usc --output usc_complete.json
python usc_vectordb_builder.py
```

The builder downloads the MiniLM model on first run (~90 MB) and writes the persistent index to `chroma_db/`.

---

## ▶️ Running the app

```bash
streamlit run app.py
```

The app exposes example scenarios and accepts free-text input. End-to-end cost on gpt-4o-mini is roughly $0.002 per analysis. Latency is roughly 90 seconds per case, dominated by Tavily search and sequential agent execution.

### Command-line mode

```bash
python main.py wire_fraud          # one of the named test cases
python main.py all                 # run all 8 test cases sequentially
```

---

## 🧪 Evaluation

The `evaluation/` directory ships a full benchmark harness covering all four agents plus cost and latency. See `evaluation/README.md` for the metric definitions. Quick start:

```bash
pip install -r evaluation/requirements-eval.txt
python -m evaluation.runners.run_e2e_eval --mock           # no API keys needed
python -m evaluation.runners.run_e2e_eval                  # real CrewAI run
```

Latest end-to-end numbers on gpt-4o-mini over 8 hand-labeled scenarios:

| Metric | Score |
|---|---|
| Retrieval Precision@1 | 1.00 |
| Retrieval MRR | 1.00 |
| Retrieval Hit-Rate@3 | 1.00 |
| Retrieval Recall@5 | 0.56 |
| Case-type classification | 1.00 |
| Legal-domain classification | 1.00 |
| Federal-hooks F1 | 0.44 |
| Drafter schema validity | 1.00 |
| Drafter citation faithfulness (no hallucinated statutes) | 1.00 |
| Drafter excerpt grounding | 1.00 |
| Drafter draft-format quality | 1.00 |
| Precedent trusted-source precision | 1.00 |
| Precedent opinion-page precision | 0.83 |
| Cases with precedents | 8/8 |
| Cost per case | ~$0.0020 |

See `evaluation/results/*.json` for the most recent run.

---

## 📂 Repository layout

```
.
├── app.py                            # Streamlit UI entry point
├── crew.py                           # CrewAI orchestration (sequential pipeline)
├── main.py                           # CLI runner with 8 named test scenarios
├── agents/                           # Four agent definitions (intake, USC, precedent, drafter)
├── tasks/                            # Task prompts that pair with each agent
├── tools/
│   ├── usc_sections_search_tool.py   # Hybrid USC retrieval + excerpt-repair post-processor
│   └── legal_precedent_search_tool.py# Tavily wrapper with trusted-domain whitelist
├── usc_parser.py                     # USC XML to JSON (run once)
├── usc_vectordb_builder.py           # JSON to ChromaDB (run once)
├── usc/                              # Source USC XML files (gitignored)
├── usc_complete.json                 # Parsed flat JSON (gitignored)
├── chroma_db/                        # Persistent ChromaDB index (gitignored)
├── evaluation/                       # Eval harness (metrics, runners, ground truth)
│   ├── data/
│   ├── metrics/
│   ├── runners/
│   ├── results/
│   └── README.md
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration reference

| Env var | Default | Effect |
|---|---|---|
| `OPENAI_API_KEY` | required | Used by all four agents. |
| `TAVILY_API_KEY` | required | Used by the precedent search tool. |
| `PERSIST_DIRECTORY_PATH` | `./chroma_db` | Where the persistent vector DB lives. |
| `USC_COLLECTION_NAME` | `usc_complete` | Chroma collection name. |
| `USC_SEARCH_TOP_K` | `8` | Top-k from the MiniLM semantic search step. |
| `USC_LEXICAL_K` | `5` | Top-k from the lexical fallback step. Merged with semantic results. |
| `TAVILY_MAX_RESULTS` | `8` | Max raw results from Tavily before filtering. |
| `TAVILY_SEARCH_DEPTH` | `advanced` | Tavily search depth. `basic` is cheaper and faster. |

---

## 🚧 Known limitations

* Federal Eagle does NOT replace a lawyer.
* The precedent agent is intentionally conservative: it returns an empty list rather than guess citations.
* The drafter only cites statutes the retriever surfaced. Citation faithfulness on the eval set is 1.00 (no hallucinated statutes), but coverage depends on the retriever finding the right sections in the first place.
* Outputs improve with more specific user input. Vague scenarios produce vague hooks and broader statute lists.
* Latency is roughly 90 seconds per case end-to-end. Most of this is Tavily search at `advanced` depth.

---

## 🦅 Why "Federal Eagle"?

The eagle represents U.S. federal authority, sharp oversight, and a high-level perspective. Federal Eagle does not argue cases, it helps you see the legal landscape clearly.

---

## 📜 License

© 2026 Rishika Mamidibathula. All rights reserved.

This project is proprietary and confidential. Use, copying, modification, or distribution is not permitted without explicit permission.
