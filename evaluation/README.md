# Federal Eagle — Evaluation Harness

This `evaluation/` package gives Federal Eagle an end-to-end evaluation suite covering all four agents plus cost / latency. It runs with **no API keys or vector DB** via a deterministic mock pipeline, then you can switch flags to evaluate the real CrewAI pipeline when keys are wired up.

## Directory layout

```
evaluation/
├── data/
│   ├── ground_truth.json         # 8 hand-labeled cases from main.py
│   ├── synthetic_generator.py    # LLM-generated extra cases (needs OPENAI_API_KEY)
│   └── legalbench_adapter.py     # Stub: HuggingFace LegalBench -> ground_truth shape
├── metrics/
│   ├── retrieval.py              # Precision@k, Recall@k, MRR, MAP, nDCG, Hit-Rate, distractor rate
│   ├── intake.py                 # JSON schema, case_type acc, legal_domain acc, federal_hooks F1, query keyword coverage
│   ├── drafter.py                # Schema, citation faithfulness, excerpt grounding, draft format quality,
│   │                             # elements-block validity, RAGAS-style faithfulness/answer-relevance/context P&R
│   ├── precedent.py              # Trusted-source precision, opinion-vs-non-opinion, dedup, no-guessing compliance, court-tier breakdown
│   └── cost_latency.py           # tiktoken-based token counting + per-stage + per-case $ and latency
├── runners/
│   ├── mock_pipeline.py          # No-keys mock that returns realistic-shaped outputs seeded from ground truth
│   ├── run_retrieval_eval.py     # Retrieval-only eval (use --mock or real chroma DB)
│   ├── run_agent_eval.py         # Case Intake eval (use --mock or real CrewAI)
│   └── run_e2e_eval.py           # Full pipeline eval (all metrics aggregated)
├── results/                      # JSON outputs land here
├── requirements-eval.txt
└── README.md
```

## What gets measured

### 1. USC Retrieval (vector search)
| Metric | What it tells you |
|---|---|
| Precision@k | Fraction of top-k that are correct statutes |
| Recall@k | Fraction of all known-correct statutes appearing in top-k |
| Hit-Rate@k | Did *any* correct statute show up in top-k? |
| MRR | Reciprocal rank of the first correct hit (1.0 = always ranks #1) |
| MAP | Mean Average Precision over all relevant statutes |
| nDCG@k | Graded relevance (primary=3, secondary=2) discounted by rank |
| Distractor rate | How often a known-misleading statute creeps into top-k |

### 2. Case Intake agent
- JSON validity (loose parse with code-fence / smart-quote tolerance)
- Schema score (required keys present, list fields are lists, `case_type` in enum)
- Case-type classification accuracy
- Legal-domain match (substring-tolerant)
- Federal-hooks soft F1 (token Jaccard ≥ 0.4 counts as match)
- Search-query keyword coverage (do generated queries contain the keywords we'd expect for this fact pattern?)

### 3. Drafter agent
- Top-level + nested schema validity
- **Citation faithfulness** — drafter only cites statutes the retriever returned (catches hallucinated statutes)
- **Excerpt grounding** — drafter excerpt is an 8-gram substring of the retrieved excerpt (catches paraphrase-as-quote)
- Draft format quality — no markdown, ALL-CAPS headings present, numbered paragraphs, placeholders used, no leaked disclaimer
- Elements-analysis validity — checklist shape + `status` in {`met`, `unknown`, `not_met`} + cites only statutes from this output
- **RAGAS-style** (no LLM judge — token-overlap proxy):
  - Faithfulness — are the drafter's claims entailed by the retrieved statute excerpts?
  - Answer Relevance — does the answer's `primary_issue` overlap with the user question?
  - Context Precision — fraction of retrieved excerpts that are actually relevant
  - Context Recall — fraction of drafter claims that *some* retrieved excerpt supports

### 4. Precedent agent
- Trusted-source precision — URLs on whitelist (law.cornell.edu, justia, courtlistener, supremecourt.gov, etc.)
- Opinion-page precision — title/citation/url contains `v.` or a reporter string (`U.S.`, `S.Ct.`, `F.3d`, `F.Supp.`)
- Dedup correctness — no near-duplicate case names
- No-guessing compliance — non-empty `court_year`/`citation` must be supported by visible text
- Court-tier breakdown — share of SCOTUS / circuit / district / unknown
- Schema validity

### 5. Cost / latency / tokens
- Per-stage wall-clock duration (case_intake, usc_retrieval, precedent_search, drafter)
- Per-stage prompt + completion tokens (via `tiktoken`, or 4-chars-per-token fallback)
- Per-case + total USD cost (pricing table in `cost_latency.py`, update as needed)

## Quickstart (no keys needed)

```bash
pip install -r evaluation/requirements-eval.txt    # just adds tiktoken
python -m evaluation.runners.run_retrieval_eval --mock
python -m evaluation.runners.run_agent_eval --mock
python -m evaluation.runners.run_e2e_eval --mock
```

Each runner writes a JSON report under `evaluation/results/`.

## Running against the real pipeline

Once you have `.env` populated and the USC chroma DB built:

```bash
# retrieval only — no LLM cost, just chroma similarity_search
python -m evaluation.runners.run_retrieval_eval --top-k 10

# intake agent only — ~$0.01/case on gpt-4o-mini
python -m evaluation.runners.run_agent_eval

# full crew (intake + USC + precedent + drafter) — ~$0.05/case on gpt-4o-mini
python -m evaluation.runners.run_e2e_eval
python -m evaluation.runners.run_e2e_eval --cases wire_fraud,bank_robbery   # subset
```

## Three ground-truth options (you asked for all three)

1. **Hand-labeled** — `data/ground_truth.json`. 8 scenarios from `main.py` annotated with expected primary/secondary statutes, distractors, case_type, federal_hooks, legal_domain, and search-query keywords. This is the "good signal" benchmark.
2. **Synthetic** — `data/synthetic_generator.py`. Run once with an OpenAI key to produce ~25–100 extra labeled cases for statistical-power runs. Lower-signal because the labels are LLM-written.
3. **LegalBench** — `data/legalbench_adapter.py`. Stub explaining how to convert HuggingFace LegalBench tasks (`citation_prediction_classification` is the closest fit) into the same ground-truth shape so the existing metrics run unchanged.

## Caveats

- The RAGAS-style metrics here are **token-overlap proxies**, not LLM-judged. For headline numbers, add an LLM-judge wrapper (Claude/GPT) on top of `faithfulness()` and `answer_relevance()`.
- The mock pipeline is seeded from the ground truth, so mock scores are **upper bounds on the real system's performance** for the parts the mock fakes. Use them only to sanity-check the metric code.
- Token counts on real runs are approximate unless you wire CrewAI's LLM callbacks into `cost_latency.StageTimer`. See the comment in `run_e2e_eval.py`.
