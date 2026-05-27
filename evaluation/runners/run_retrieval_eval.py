"""Evaluate the USC retrieval step in isolation.

Usage:
    # Mock (no API keys / no chroma DB needed):
    python -m evaluation.runners.run_retrieval_eval --mock

    # Real (requires built chroma_db; runs entirely offline, no LLM calls):
    python -m evaluation.runners.run_retrieval_eval --top-k 10

Writes JSON report to evaluation/results/retrieval.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from ..metrics.retrieval import evaluate_run
from .mock_pipeline import mock_usc_retrieval


ROOT = Path(__file__).resolve().parents[2]
GT_PATH = ROOT / "evaluation" / "data" / "ground_truth.json"
OUT_PATH = ROOT / "evaluation" / "results" / "retrieval.json"


def real_retrieval(case, top_k: int) -> List[str]:
    """Call the project's real Chroma-backed retriever. Each ground-truth case
    is run against ALL of its expected search-query keywords; results are merged
    in rank order (first-seen wins), then truncated to top_k.
    """
    from tools.usc_sections_search_tool import search_usc_sections  # type: ignore

    seen = []
    for q in case.get("expected_search_query_keywords", []) or [case["name"]]:
        results = search_usc_sections.func(q) if hasattr(search_usc_sections, "func") else search_usc_sections(q)
        for r in results or []:
            c = r.get("citation")
            if c and c not in seen:
                seen.append(c)
        if len(seen) >= top_k:
            break
    return seen[:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="Use mock retriever (no chroma DB needed)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cases = json.loads(GT_PATH.read_text())["cases"]
    runs = []
    for c in cases:
        if args.mock:
            retrieved = mock_usc_retrieval(c, k=args.top_k, seed=args.seed)
        else:
            retrieved = real_retrieval(c, top_k=args.top_k)
        runs.append({
            "case_id": c["id"],
            "retrieved": retrieved,
            "expected_statutes": c["expected_statutes"],
        })

    summary = evaluate_run(runs, ks=(1, 3, 5, 10))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"summary": summary, "per_case": runs}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nFull report -> {OUT_PATH}")


if __name__ == "__main__":
    main()
