"""Token / cost / latency instrumentation.

Usage:
    from evaluation.metrics.cost_latency import StageTimer, count_tokens, estimate_cost

    with StageTimer("case_intake") as t:
        out = run_intake(...)
    t.tokens(prompt=count_tokens(prompt, "gpt-4o-mini"), completion=count_tokens(out, "gpt-4o-mini"))

Rates below are USD per 1M tokens. Update as needed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Dict, List, Optional


# USD per 1M tokens (as of mid-2025; update if outdated)
PRICING = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4.1-mini": {"prompt": 0.40, "completion": 1.60},
    "gpt-4.1": {"prompt": 2.00, "completion": 8.00},
    "claude-3-5-sonnet-latest": {"prompt": 3.00, "completion": 15.00},
    "claude-3-5-haiku-latest": {"prompt": 0.80, "completion": 4.00},
}


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Use tiktoken if installed; otherwise fall back to a 4-chars-per-token estimate."""
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rate = PRICING.get(model)
    if not rate:
        return 0.0
    return (prompt_tokens / 1_000_000) * rate["prompt"] + (completion_tokens / 1_000_000) * rate["completion"]


@dataclass
class StageMetric:
    stage: str
    duration_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "gpt-4o-mini"
    cost_usd: float = 0.0


class StageTimer:
    """Context manager that measures wall-clock time for a pipeline stage."""

    def __init__(self, stage: str, model: str = "gpt-4o-mini"):
        self.metric = StageMetric(stage=stage, model=model)
        self._t0: Optional[float] = None

    def __enter__(self) -> "StageTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._t0 is not None:
            self.metric.duration_s = time.perf_counter() - self._t0

    def tokens(self, prompt: int = 0, completion: int = 0):
        self.metric.prompt_tokens += prompt
        self.metric.completion_tokens += completion
        self.metric.cost_usd = estimate_cost(self.metric.model, self.metric.prompt_tokens, self.metric.completion_tokens)


@dataclass
class CaseRunMetrics:
    case_id: str
    stages: List[StageMetric] = field(default_factory=list)

    def add(self, m: StageMetric) -> None:
        self.stages.append(m)

    def total_duration(self) -> float:
        return sum(s.duration_s for s in self.stages)

    def total_tokens(self) -> Dict[str, int]:
        return {
            "prompt": sum(s.prompt_tokens for s in self.stages),
            "completion": sum(s.completion_tokens for s in self.stages),
        }

    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.stages)

    def to_dict(self) -> Dict:
        return {
            "case_id": self.case_id,
            "total_duration_s": self.total_duration(),
            "total_tokens": self.total_tokens(),
            "total_cost_usd": self.total_cost(),
            "stages": [s.__dict__ for s in self.stages],
        }


def aggregate_cost_latency(runs: List[CaseRunMetrics]) -> Dict:
    if not runs:
        return {}
    totals_dur = [r.total_duration() for r in runs]
    totals_cost = [r.total_cost() for r in runs]
    totals_p = [r.total_tokens()["prompt"] for r in runs]
    totals_c = [r.total_tokens()["completion"] for r in runs]
    by_stage: Dict[str, List[StageMetric]] = {}
    for r in runs:
        for s in r.stages:
            by_stage.setdefault(s.stage, []).append(s)
    return {
        "n_cases": len(runs),
        "duration_s": {"mean": mean(totals_dur), "median": median(totals_dur), "max": max(totals_dur)},
        "cost_usd": {"mean": mean(totals_cost), "median": median(totals_cost), "total": sum(totals_cost)},
        "tokens": {
            "prompt_mean": mean(totals_p),
            "completion_mean": mean(totals_c),
            "total_prompt": sum(totals_p),
            "total_completion": sum(totals_c),
        },
        "per_stage_duration_s_mean": {
            stage: mean(s.duration_s for s in lst) for stage, lst in by_stage.items()
        },
        "per_stage_cost_usd_mean": {
            stage: mean(s.cost_usd for s in lst) for stage, lst in by_stage.items()
        },
    }
