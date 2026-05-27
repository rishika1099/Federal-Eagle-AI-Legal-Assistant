"""Evaluate the Case Intake agent in isolation.

Usage:
    python -m evaluation.runners.run_agent_eval --mock
    python -m evaluation.runners.run_agent_eval         # uses real CrewAI agent (needs OPENAI_API_KEY)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..metrics.intake import evaluate_intake_run
from .mock_pipeline import mock_intake


ROOT = Path(__file__).resolve().parents[2]
GT_PATH = ROOT / "evaluation" / "data" / "ground_truth.json"
OUT_PATH = ROOT / "evaluation" / "results" / "intake.json"


def real_intake(case):
    """Run only the case_intake agent. Imports late so --mock doesn't require crewai."""
    from crewai import Crew  # type: ignore
    from agents.case_intake_agent import case_intake_agent  # type: ignore
    from tasks.case_intake_task import case_intake_task  # type: ignore

    crew = Crew(agents=[case_intake_agent], tasks=[case_intake_task], process="sequential", verbose=False)
    res = crew.kickoff(inputs={"user_input": case["scenario"]})
    return str(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cases = json.loads(GT_PATH.read_text())["cases"]
    runs = []
    for c in cases:
        predicted = mock_intake(c, args.seed) if args.mock else real_intake(c)
        runs.append({"case_id": c["id"], "predicted": predicted, "expected": c})

    summary = evaluate_intake_run(runs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"summary": summary, "per_case_predictions": [r["predicted"] for r in runs]}, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    print(f"\nFull report -> {OUT_PATH}")


if __name__ == "__main__":
    main()
