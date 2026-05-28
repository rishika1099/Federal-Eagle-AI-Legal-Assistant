"""Safety gates that run BEFORE the crew kicks off.

Three layers, in order:
  1) Hard-blocked topic regex (minors as victims, self-harm/suicide,
     active criminal procedure such as ongoing prosecution, weapons-of-mass-
     destruction synthesis, etc.). These never reach an LLM.
  2) Prompt-injection / jailbreak sanitizer. Detects classic prompt-overrides
     in the user input and either rejects or strips them.
  3) OpenAI Moderation API call. Catches the categories we don't enumerate.

Each gate returns SafetyDecision(allowed, reason, category, sanitized_input).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

# Patterns are intentionally over-broad. Better to false-positive-block a
# borderline query than to publish a tool that helps with the bad stuff.
_HARD_BLOCK_PATTERNS = [
    (re.compile(r"\b(child|minor|infant|underage)\b.*\b(abuse|sexual|porn|trafficking|exploit)\b", re.I), "minor_protection"),
    (re.compile(r"\b(sexual|porn|nude)\b.*\b(child|minor|underage|kid)\b", re.I), "minor_protection"),
    (re.compile(r"\b(suicide|kill\s+myself|end\s+my\s+life|self[- ]harm)\b", re.I), "self_harm"),
    (re.compile(r"\bhow\s+(do\s+i|to|can\s+i)\b.*\b(make|build|synthesize|manufacture)\b.*\b(bomb|explosive|nerve\s+agent|sarin|ricin|anthrax|nuclear)\b", re.I), "wmd"),
    (re.compile(r"\b(jailbreak|bypass)\b.*\b(disclaimer|safety|filter|guardrail|moderation)\b", re.I), "jailbreak_explicit"),
]

# Soft-detect prompt injection. We don't auto-block; we strip the matching
# spans from the user input before sending to the agents.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"disregard\s+(the|your|all)\s+(rules|instructions|prompt)", re.I),
    re.compile(r"you\s+are\s+now\s+([A-Z][A-Za-z]+|a\s+(jailbroken|unrestricted)\b)", re.I),
    re.compile(r"system\s*[:>]\s*", re.I),
    re.compile(r"### system\b", re.I),
    re.compile(r"do\s+not\s+(follow|obey)\s+the", re.I),
    re.compile(r"act\s+as\s+(dan|developer\s+mode)", re.I),
]


@dataclass
class SafetyDecision:
    allowed: bool
    sanitized_input: str
    category: Optional[str] = None  # which gate fired
    reason: Optional[str] = None
    moderation_categories: Optional[dict] = None  # raw moderation API response

    def as_log_payload(self) -> dict:
        return {
            "allowed": self.allowed,
            "category": self.category,
            "reason": self.reason,
            "had_injection": self.sanitized_input is not None,
            "moderation_flagged_categories": [
                k for k, v in (self.moderation_categories or {}).items() if v
            ] if self.moderation_categories else [],
        }


def _hard_block_check(text: str) -> Optional[Tuple[str, str]]:
    """Return (category, matched_excerpt) if text hits a hard-block pattern."""
    for pat, category in _HARD_BLOCK_PATTERNS:
        m = pat.search(text)
        if m:
            return category, m.group(0)[:80]
    return None


def _strip_injection(text: str) -> Tuple[str, bool]:
    """Return (cleaned, had_injection). Removes matched injection spans."""
    out = text
    had = False
    for pat in _INJECTION_PATTERNS:
        new = pat.sub("[redacted prompt-injection attempt]", out)
        if new != out:
            had = True
        out = new
    return out, had


def _moderation_check(text: str) -> Optional[dict]:
    """Call OpenAI moderation API. Returns the categories dict or None on failure.
    We treat ANY flagged category as a soft warning unless it's in the hard set below.
    """
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI()
        resp = client.moderations.create(
            model=os.getenv("FEDERAL_EAGLE_MOD_MODEL", "omni-moderation-latest"),
            input=text[:4000],
        )
        if not resp.results:
            return None
        r = resp.results[0]
        cats = r.categories.model_dump() if hasattr(r.categories, "model_dump") else dict(r.categories)
        return cats
    except Exception:
        return None


_MOD_HARD_BLOCK = {"sexual/minors", "violence/graphic", "self-harm/intent", "self-harm/instructions"}


def check_input(user_input: str) -> SafetyDecision:
    """Run all three gates. Order: hard regex → injection sanitizer → moderation API."""
    text = (user_input or "").strip()
    if not text:
        return SafetyDecision(allowed=False, sanitized_input="", category="empty", reason="empty input")

    # Layer 1: hard topic block.
    hit = _hard_block_check(text)
    if hit:
        category, excerpt = hit
        return SafetyDecision(
            allowed=False,
            sanitized_input=text,
            category=category,
            reason=f"Hard-block category '{category}' matched. Federal Eagle does not analyze this topic.",
        )

    # Layer 2: prompt-injection stripping.
    cleaned, had_injection = _strip_injection(text)

    # Layer 3: OpenAI moderation API. Soft signal, hard-block only on a small set.
    mod_categories = _moderation_check(cleaned)
    if mod_categories:
        flagged = [k for k, v in mod_categories.items() if v]
        if any(k in _MOD_HARD_BLOCK for k in flagged):
            return SafetyDecision(
                allowed=False,
                sanitized_input=cleaned,
                category="moderation_block",
                reason=f"Moderation API flagged hard-block category: {[k for k in flagged if k in _MOD_HARD_BLOCK]}",
                moderation_categories=mod_categories,
            )

    return SafetyDecision(
        allowed=True,
        sanitized_input=cleaned,
        category="injection_redacted" if had_injection else None,
        reason="injection patterns redacted" if had_injection else None,
        moderation_categories=mod_categories,
    )
