"""LegalBench adapter (STUB).

LegalBench (https://hazyresearch.stanford.edu/legalbench/) is a public legal-reasoning
benchmark with ~160 tasks. Most are short classification / rule-application tasks rather
than long fact-pattern -> statute lookup, so only a subset maps cleanly onto Federal Eagle.

Recommended LegalBench sub-tasks for this project:
  - `citation_prediction_classification` and `citation_prediction_open` — closest to
    USC retrieval (given a holding, predict the cited statute/case).
  - `definition_classification` and `definition_extraction` — relevant to elements analysis.
  - `rule_qa` — closest to the drafter's "why_relevant" claims.

How to use (manual, since LegalBench is a HuggingFace dataset):

    pip install datasets
    from datasets import load_dataset
    ds = load_dataset("nguha/legalbench", "citation_prediction_classification", split="test")

Then convert each row into the ground_truth.json shape:
    {
      "id": f"lb_citation_{row['idx']}",
      "name": "LegalBench citation prediction",
      "scenario": row["text"],
      "expected_case_type": "unclear",
      "expected_legal_domain": ["Citation Prediction"],
      "expected_federal_hooks": [],
      "expected_statutes": {"primary": [row["answer"]], "secondary": [], "distractor_avoid": []},
      "expected_search_query_keywords": []
    }

This file is intentionally a stub so the eval harness compiles. Implement the loader
when you're ready — the existing metric modules will work unchanged on the converted rows.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "evaluation" / "data" / "legalbench.json"


def convert_row(row: dict, task: str, idx: int) -> dict:
    return {
        "id": f"lb_{task}_{idx}",
        "name": f"LegalBench {task}",
        "scenario": row.get("text", "") or row.get("question", ""),
        "expected_case_type": "unclear",
        "expected_legal_domain": [task.replace("_", " ").title()],
        "expected_federal_hooks": [],
        "expected_statutes": {
            "primary": [row["answer"]] if "answer" in row else [],
            "secondary": [],
            "distractor_avoid": [],
        },
        "expected_search_query_keywords": [],
    }


def main():
    raise SystemExit(
        "Stub. Install `datasets`, load_dataset('nguha/legalbench', <task>), iterate, "
        "and call convert_row(...) to materialize evaluation/data/legalbench.json. "
        "Then point any runner at it the same way it reads ground_truth.json."
    )


if __name__ == "__main__":
    main()
