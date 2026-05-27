"""Generate synthetic labeled eval scenarios with an LLM.

Use this AFTER you have an OpenAI key set in the environment. It produces
additional cases in the same shape as ground_truth.json so the existing
metrics work unchanged.

WARNING: LLM-generated labels are a weaker signal than hand labels. Use
synthetic data for trend tracking / variance estimation, not for headline numbers.

Usage:
    python -m evaluation.data.synthetic_generator --n 25 --seed-cases identity_theft,wire_fraud
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
