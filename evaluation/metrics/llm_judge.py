"""LLM-judged RAGAS-style metrics.

Replaces the token-overlap proxies in drafter.py with actual model-graded
judgments. Uses gpt-4o-mini at temperature 0 and JSON-mode for determinism.

Three judges, all returning floats in [0, 1]:

  - judge_faithfulness(claims, contexts):
      Of the drafter's claims, what fraction are entailed by the retrieved
      statute text? An "unfaithful" claim is one that says something the
      statutes don't actually say.

  - judge_answer_relevance(question, answer):
      How well does the answer address the question, ignoring whether it's
      legally correct? Catches off-topic outputs.

  - judge_context_recall(answer_claims, contexts):
      Of the things the answer claims, what fraction are findable in the
      retrieved contexts? Lower recall = the answer is using info from outside
      the retrieval (hallucination risk).

Cost: each judge call is ~$0.0005 on gpt-4o-mini. Three judges per case.
Run only when you want headline numbers, not on every eval.
"""
from __future__ import annotations

import json
import os
from statistics import mean
from typing import Dict, List, Optional, Sequence

_JUDGE_MODEL = os.getenv("FEDERAL_EAGLE_JUDGE_MODEL", "gpt-4o-mini")
_JUDGE_TEMPERATURE = float(os.getenv("FEDERAL_EAGLE_JUDGE_TEMP", "0"))


def _client():
    from openai import OpenAI  # type: ignore
    return OpenAI()


def _ask_json(system: str, user: str) -> Optional[dict]:
    """Single JSON-mode call. Returns parsed dict or None on failure."""
    try:
        resp = _client().chat.completions.create(
            model=_JUDGE_MODEL,
            temperature=_JUDGE_TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

_FAITHFULNESS_SYSTEM = (
    "You are a strict legal evaluator. Decide whether each claim is supported by "
    "the provided statute excerpts. A claim is supported only if the excerpts "
    "directly entail it. If the excerpts are silent, the claim is NOT supported. "
    "Return JSON: {\"judgments\": [{\"claim\": \"...\", \"supported\": true|false, \"reason\": \"...\"}]}."
)


def judge_faithfulness(claims: Sequence[str], contexts: Sequence[str]) -> Dict[str, float]:
    """Fraction of claims entailed by contexts. Returns {'faithfulness': float, 'n_claims': int}."""
    claims = [c for c in claims if c]
    if not claims:
        return {"faithfulness": 0.0, "n_claims": 0}
    user = (
        "STATUTE EXCERPTS:\n"
        + "\n---\n".join(contexts or ["(none)"])
        + "\n\nCLAIMS TO JUDGE:\n"
        + json.dumps(list(claims))
    )
    parsed = _ask_json(_FAITHFULNESS_SYSTEM, user)
    judgments = (parsed or {}).get("judgments") or []
    if not judgments:
        return {"faithfulness": 0.0, "n_claims": len(claims)}
    supported = sum(1 for j in judgments if j.get("supported") is True)
    return {"faithfulness": supported / len(judgments), "n_claims": len(judgments)}


# ---------------------------------------------------------------------------
# Answer relevance
# ---------------------------------------------------------------------------

_RELEVANCE_SYSTEM = (
    "Rate, on a 0.0 to 1.0 scale, how well the ANSWER addresses the QUESTION. "
    "Ignore whether the answer is legally correct. 1.0 = directly answers. "
    "0.5 = partially on topic. 0.0 = off topic. "
    "Return JSON: {\"relevance\": <float>, \"reason\": \"...\"}."
)


def judge_answer_relevance(question: str, answer: str) -> Dict[str, float]:
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
    parsed = _ask_json(_RELEVANCE_SYSTEM, user) or {}
    raw = parsed.get("relevance", 0.0)
    try:
        score = max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        score = 0.0
    return {"answer_relevance": score}


# ---------------------------------------------------------------------------
# Context recall
# ---------------------------------------------------------------------------

_CONTEXT_RECALL_SYSTEM = (
    "For each claim in the ANSWER, decide whether the supporting information can "
    "be found IN the provided CONTEXTS. This is a recall check: would a reader "
    "with only the contexts be able to derive the claim? "
    "Return JSON: {\"judgments\": [{\"claim\": \"...\", \"found_in_context\": true|false}]}."
)


def judge_context_recall(answer_claims: Sequence[str], contexts: Sequence[str]) -> Dict[str, float]:
    claims = [c for c in answer_claims if c]
    if not claims:
        return {"context_recall": 0.0, "n_claims": 0}
    user = (
        "CONTEXTS:\n"
        + "\n---\n".join(contexts or ["(none)"])
        + "\n\nANSWER CLAIMS:\n"
        + json.dumps(list(claims))
    )
    parsed = _ask_json(_CONTEXT_RECALL_SYSTEM, user) or {}
    judgments = parsed.get("judgments") or []
    if not judgments:
        return {"context_recall": 0.0, "n_claims": len(claims)}
    found = sum(1 for j in judgments if j.get("found_in_context") is True)
    return {"context_recall": found / len(judgments), "n_claims": len(judgments)}


# ---------------------------------------------------------------------------
# Aggregation over a list of cases
# ---------------------------------------------------------------------------

def evaluate_llm_judged(runs: List[Dict]) -> Dict[str, float]:
    """runs: [{case_id, question, answer, claims, contexts}]"""
    faith, rel, recall = [], [], []
    for r in runs:
        faith.append(judge_faithfulness(r.get("claims", []), r.get("contexts", []))["faithfulness"])
        rel.append(judge_answer_relevance(r.get("question", ""), r.get("answer", ""))["answer_relevance"])
        recall.append(judge_context_recall(r.get("claims", []), r.get("contexts", []))["context_recall"])
    return {
        "n_cases": len(runs),
        "llm_judge_faithfulness": mean(faith) if faith else 0.0,
        "llm_judge_answer_relevance": mean(rel) if rel else 0.0,
        "llm_judge_context_recall": mean(recall) if recall else 0.0,
    }
