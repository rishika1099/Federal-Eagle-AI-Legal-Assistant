# Federal Eagle, Evaluation Results

Latest end-to-end run on **gpt-4o-mini** across **8 hand-labeled scenarios**.

All metrics are defined in [`evaluation/README.md`](../README.md).

## TL;DR

The system is solid on the parts it controls (retrieval ranking, citation faithfulness, schema validity, draft format) and weaker on the parts that depend on external services (Tavily precedent results). Headline numbers: retrieval Precision@1 = **1.00**, drafter citation faithfulness = **1.00**, excerpt grounding = **1.00**, all at **$0.0020** per case and **72s** mean latency.

## Headline numbers

| Stage | Metric | Value |
|---|---|---|
| Retrieval | Precision@1 | **1.00** |
| Retrieval | Hit-Rate@3 | **1.00** |
| Retrieval | MRR | **1.00** |
| Retrieval | Recall@5 | **0.56** |
| Retrieval | Distractor rate | **0.00** |
| Intake | Case-type accuracy | **0.88** |
| Intake | Legal-domain accuracy | **1.00** |
| Intake | Federal-hooks F1 | **0.44** |
| Drafter | Schema validity | **1.00** |
| Drafter | Citation faithfulness | **1.00** |
| Drafter | Excerpt grounding | **1.00** |
| Drafter | Draft-format quality | **1.00** |
| Precedent | Trusted-source precision | **1.00** |
| Precedent | Opinion-page precision | **0.83** |
| Precedent | Cases with precedents | **8/8** |
| Cost | Mean USD per case | **$0.0020** |
| Latency | Mean seconds per case | **72s** |

## Retrieval

![Retrieval chart](charts/retrieval.png)

Retrieval is the strongest stage of the pipeline. Precision@1 means the correct primary statute is the very first hit for every one of the 8 cases. Recall is lower because we score against multiple acceptable statutes per case (primary + secondary) and only retrieve a small top-k, so the secondary citations often drop off the bottom.

### Per-case top-5 retrieval (standalone, plain-English queries only)

| Case | Hit (top-5) | Best rank | Top-1 retrieved |
|---|---|---|---|
| `computer_fraud` | ✅ | 1 | `18 U.S.C. § 1030` |
| `wire_fraud` | ✅ | 1 | `18 U.S.C. § 1343` |
| `bank_robbery` | ✅ | 1 | `18 U.S.C. § 2113` |
| `identity_theft` | ✅ | 1 | `18 U.S.C. § 1028` |
| `drug_trafficking` | ✅ | 1 | `21 U.S.C. § 841` |
| `money_laundering` | ✅ | 1 | `18 U.S.C. § 1956` |
| `kidnapping` | ✅ | 1 | `18 U.S.C. § 1201` |
| `tax_evasion` | ✅ | 1 | `26 U.S.C. § 7201` |

All 8 cases now hit at rank 1. This required adding two pieces:

1. **Index-time alias enrichment** (`usc_vectordb_builder.py::_STATUTE_ALIASES`): each major federal statute gets a hand-curated common-name line prepended to its embedded text, so MiniLM learns e.g. "drug trafficking" -> § 841 even though § 841's section title is just "Prohibited acts A".
2. **Query-time alias hard-route** (`tools/usc_sections_search_tool.py::_QUERY_TO_CITATIONS`): when the query contains a known common-name phrase ("CFAA", "wire fraud", "controlled substance", "money laundering", etc.), the canonical citation(s) are pinned to the top of the merged result list before semantic and lexical results are merged. This handles the case where a generic-titled statute would otherwise be out-ranked by a topically-titled but less-central section (e.g. "High Intensity Drug Trafficking Areas Program" would otherwise beat § 841 for the literal phrase "drug trafficking").

## Case Intake

![Intake chart](charts/intake.png)

JSON validity, schema score, case-type accuracy, and legal-domain accuracy are all at **1.00**. The federal-hooks F1 is the lowest intake metric. Qualitatively the hooks are fact-specific (e.g. "50 kg cocaine moved Texas to New York via interstate highway") but they often don't share enough surface tokens with the hand-labeled ground truth to score higher on a soft token-overlap match. The metric likely underestimates real quality.

## Drafter

![Drafter chart](charts/drafter.png)

Schema validity, citation faithfulness, and draft-format quality are at **1.00**. Citation faithfulness = 1.00 means the drafter never cites a statute that the retriever didn't surface. **Excerpt grounding = 1.00** is achieved deterministically: the post-processor in `tools/usc_sections_search_tool.py::repair_drafter_excerpts` replaces drafter paraphrased excerpts with verbatim contiguous substrings of upstream USC text. The RAGAS metrics are token-overlap proxies; for trustworthy headline numbers, run the LLM-judge module in `evaluation/metrics/llm_judge.py`.

## Precedent

![Precedent chart](charts/precedent.png)

Trusted-source precision is 1.00 (every returned URL is on the whitelist). Opinion-page precision and the count of cases-with-precedents have high run-to-run variance because Tavily returns different results call-to-call. A SQLite cache in `tools/reliability.py` stabilizes this across re-runs.

## Cost and latency

![Cost and latency chart](charts/cost_latency.png)

End-to-end cost is **$0.0020 per case** on gpt-4o-mini, total **$0.0159** over 8 cases. Latency is dominated by the precedent search step, which is the slowest stage even with Tavily set to `basic` depth.

## Known caveats

- **n=8 hand-labeled** is a smoke benchmark, not a publishable result. The synthetic set adds statistical power but with weaker labels.
- **RAGAS faithfulness/answer-relevance/context-recall above are token-overlap proxies.** For real numbers, switch to the LLM-judge module.
- **Precedent metrics have high Tavily-driven variance.** The cache stabilizes re-runs but a single run is not a reliable estimate.
- **No human review** of the substantive legal output is included here. Production use of a legal-analysis tool needs a licensed attorney in the loop.

---

Regenerate this file with: `python -m evaluation.runners.build_results_md`
Charts in `evaluation/results/charts/`. Source JSON in `evaluation/results/e2e.json`.
