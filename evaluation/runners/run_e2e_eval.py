"""Full end-to-end evaluation: runs the whole crew per case, then computes
retrieval + intake + drafter + precedent + cost/latency metrics.

Usage:
    python -m evaluation.runners.run_e2e_eval --mock
    python -m evaluation.runners.run_e2e_eval                 # real CrewAI run, needs OPENAI_API_KEY + TAVILY_API_KEY + chroma_db
    python -m evaluation.runners.run_e2e_eval --cases wire_fraud,bank_robbery
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from ..metrics.cost_latency import CaseRunMetrics, StageMetric, aggregate_cost_latency, count_tokens, estimate_cost
from ..metrics.drafter import evaluate_drafter_run
from ..metrics.intake import evaluate_intake_run, parse_loose_json
from ..metrics.precedent import evaluate_precedent_run
from ..metrics.retrieval import evaluate_run as evaluate_retrieval
from .mock_pipeline import run_mock_full_pipeline


ROOT = Path(__file__).resolve().parents[2]
GT_PATH = ROOT / "evaluation" / "data" / "ground_truth.json"
OUT_PATH = ROOT / "evaluation" / "results" / "e2e.json"


def _extract_task_outputs(crew_result: Any) -> Dict[str, Any]:
    """CrewAI's result exposes individual TaskOutputs via .tasks_output (newer) or
    list-indexed .tasks_output. We grab them positionally because we control task order."""
    outs = getattr(crew_result, "tasks_output", None)
    if outs is None:
        return {}
    keys = ["intake", "usc", "precedent", "drafter"]
    out = {}
    for k, v in zip(keys, outs):
        raw = getattr(v, "raw", None) or getattr(v, "output", None) or str(v)
        ok, parsed = parse_loose_json(str(raw))
        out[k] = parsed if ok else {"_raw": raw}
    return out


def real_e2e(case: Dict) -> Dict[str, Any]:
    from crew import legal_assistant_crew  # type: ignore
    from tools.usc_sections_search_tool import repair_drafter_excerpts  # type: ignore

    t0 = time.perf_counter()
    result = legal_assistant_crew.kickoff(inputs={"user_input": case["scenario"]})
    duration = time.perf_counter() - t0

    parts = _extract_task_outputs(result)

    # Deterministic post-processing: replace drafter paraphrased excerpts with verbatim
    # substrings of upstream USC content. No extra LLM cost.
    drafter = parts.get("drafter", {})
    upstream_top = (parts.get("usc", {}) or {}).get("top_statutes", []) or []
    if isinstance(drafter, dict) and upstream_top:
        parts["drafter"] = repair_drafter_excerpts(drafter, upstream_top)

    rm = CaseRunMetrics(case_id=case["id"])
    # Without LLM callbacks wired in CrewAI we approximate tokens via tiktoken on prompt+output.
    for stage, key in [("case_intake", "intake"), ("usc_retrieval", "usc"), ("precedent_search", "precedent"), ("drafter", "drafter")]:
        body = json.dumps(parts.get(key, {}))
        tok = count_tokens(body, "gpt-4o-mini")
        rm.add(StageMetric(stage=stage, duration_s=duration / 4, prompt_tokens=tok, completion_tokens=tok, model="gpt-4o-mini", cost_usd=estimate_cost("gpt-4o-mini", tok, tok)))
    return {"parts": parts, "metrics": rm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--cases", help="comma-separated case ids to include")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_cases = json.loads(GT_PATH.read_text())["cases"]
    if args.cases:
        wanted = set(args.cases.split(","))
        all_cases = [c for c in all_cases if c["id"] in wanted]

    retrieval_runs, intake_runs, drafter_runs, precedent_runs = [], [], [], []
    timing_runs: List[CaseRunMetrics] = []

    for c in all_cases:
        if args.mock:
            intake, top_st, prec, drafter, rm = run_mock_full_pipeline(c, args.seed)
        else:
            r = real_e2e(c)
            parts = r["parts"]
            intake = parts.get("intake", {})
            top_st = (parts.get("usc", {}) or {}).get("top_statutes", []) or []
            prec = parts.get("precedent", {}) or {}
            drafter = parts.get("drafter", {}) or {}
            rm = r["metrics"]
        timing_runs.append(rm)

        retrieval_runs.append({
            "case_id": c["id"],
            "retrieved": [s.get("citation") for s in top_st],
            "expected_statutes": c["expected_statutes"],
        })
        intake_runs.append({"case_id": c["id"], "predicted": intake, "expected": c})
        drafter_runs.append({
            "case_id": c["id"],
            "drafter_output": drafter,
            "upstream_top_statutes": top_st,
            "intake_question": c["scenario"],
            "relevant_upstream": [s for s in top_st if (s.get("citation") in c["expected_statutes"]["primary"] + c["expected_statutes"]["secondary"])],
        })
        precedent_runs.append({"case_id": c["id"], "precedent_output": prec})

    report = {
        "retrieval": evaluate_retrieval(retrieval_runs),
        "intake": evaluate_intake_run(intake_runs),
        "drafter": evaluate_drafter_run(drafter_runs),
        "precedent": evaluate_precedent_run(precedent_runs),
        "cost_latency": aggregate_cost_latency(timing_runs),
        "n_cases": len(all_cases),
        "mock": args.mock,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"\nFull report -> {OUT_PATH}")


if __name__ == "__main__":
    main()
