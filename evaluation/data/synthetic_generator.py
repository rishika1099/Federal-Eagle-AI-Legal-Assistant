"""Generate synthetic labeled eval scenarios with an LLM, or adapt LegalBench rows.

Two ground-truth options live here so they share the same output shape as
evaluation/data/ground_truth.json.

A) SYNTHETIC LLM GENERATION (this script's default behaviour)
   Use this AFTER you have an OpenAI key set in the environment.
   WARNING: LLM-generated labels are a weaker signal than hand labels. Use
   synthetic data for trend tracking / variance estimation, not for headline numbers.

   Usage:
       python -m evaluation.data.synthetic_generator --n 25 --seed-cases identity_theft,wire_fraud

B) LEGALBENCH ADAPTER  (call convert_legalbench_row from a small script)
   LegalBench (https://hazyresearch.stanford.edu/legalbench/) has ~160 tasks.
   Most are short classification rather than fact-pattern -> statute lookup, so
   only a subset maps cleanly. Best fits for Federal Eagle:
     - `citation_prediction_classification` / `citation_prediction_open`
       (closest to USC retrieval: given a holding, predict the cited statute)
     - `definition_classification` / `definition_extraction` (elements analysis)
     - `rule_qa` (drafter's why_relevant claims)

   Example loader:
       from datasets import load_dataset
       from evaluation.data.synthetic_generator import convert_legalbench_row
       ds = load_dataset("nguha/legalbench", "citation_prediction_classification", split="test")
       cases = [convert_legalbench_row(r, "citation_prediction_classification", i)
                for i, r in enumerate(ds)]
       # write cases to evaluation/data/legalbench.json — same shape as ground_truth.json.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GT_PATH = ROOT / "evaluation" / "data" / "ground_truth.json"
OUT_PATH = ROOT / "evaluation" / "data" / "synthetic.json"


SYSTEM = """You generate hypothetical, fictional U.S. federal-law scenarios for benchmarking an
information-retrieval system. Each scenario must be:
- 3-5 sentences of plain English
- contain at least one clear federal hook (interstate conduct, federal agency, federal property, federally insured institution, SSN/identity docs, controlled substance, IRS, etc.)
- map to one PRIMARY U.S. Code section that you state explicitly

Return ONLY valid JSON, a list of objects each with keys:
  id (snake_case unique), name, scenario, expected_case_type (criminal|civil|administrative|unclear),
  expected_legal_domain (list of 1-3 short strings), expected_federal_hooks (list of strings),
  expected_statutes (object with keys primary, secondary, distractor_avoid; each a list of citations like '18 U.S.C. § 1030'),
  expected_search_query_keywords (list of short strings)
"""


def generate(n: int, seeds: list[str]) -> list[dict]:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        raise SystemExit("Install `openai` first: pip install openai")
    client = OpenAI()
    examples = []
    if seeds:
        existing = {c["id"]: c for c in json.loads(GT_PATH.read_text())["cases"]}
        examples = [existing[s] for s in seeds if s in existing]
    msg = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Generate {n} new diverse scenarios. Avoid duplicating these existing ids: {[e['id'] for e in examples]}. Reference style examples:\n{json.dumps(examples[:2], indent=2)}"},
    ]
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=msg, temperature=0.7, response_format={"type": "json_object"})
    txt = resp.choices[0].message.content
    data = json.loads(txt)
    if isinstance(data, dict):
        # try to find the list
        for v in data.values():
            if isinstance(v, list):
                return v
        return [data]
    return data


def convert_legalbench_row(row: dict, task: str, idx: int) -> dict:
    """Convert one LegalBench HuggingFace row to ground_truth.json case-shape."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed-cases", default="wire_fraud,identity_theft")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to generate synthetic data.")
    cases = generate(args.n, args.seed_cases.split(",") if args.seed_cases else [])
    OUT_PATH.write_text(json.dumps({"schema_version": "1.0", "synthetic": True, "cases": cases}, indent=2))
    print(f"Wrote {len(cases)} synthetic cases -> {OUT_PATH}")


if __name__ == "__main__":
    main()
