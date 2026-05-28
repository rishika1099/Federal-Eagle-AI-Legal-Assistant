"""Reliability primitives: kill switch, vendor fallback, rate limiter, Tavily cache.

All four are intentionally small. None of them try to be a real platform; they
exist so the app can fail safely and predictably in a public-ish deployment.

  - is_enabled():               feature flag from FEDERAL_EAGLE_ENABLED.
  - choose_llm_provider():      OpenAI by default; Anthropic if FEDERAL_EAGLE_PROVIDER=anthropic
                                or if OpenAI health-check fails.
  - SessionRateLimiter:         per-key call cap with a sliding window.
  - TavilyCache:                sqlite-backed cache so identical queries return
                                identical results (stabilizes precedent agent).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """Master kill switch. Default ON. Set FEDERAL_EAGLE_ENABLED=false to disable."""
    return os.getenv("FEDERAL_EAGLE_ENABLED", "true").lower() not in ("false", "0", "off")


def disabled_message() -> str:
    return os.getenv(
        "FEDERAL_EAGLE_DISABLED_MESSAGE",
        "Federal Eagle is temporarily disabled. Please try again later.",
    )


# ---------------------------------------------------------------------------
# Vendor selection / LLM fallback
# ---------------------------------------------------------------------------

@dataclass
class LLMChoice:
    provider: str               # "openai" | "anthropic"
    model: str
    fallback_used: bool = False
    fallback_reason: Optional[str] = None


def choose_llm_provider(default_model_openai: str = "gpt-4o-mini",
                        default_model_anthropic: str = "claude-3-5-haiku-latest") -> LLMChoice:
    """Pick a provider, with a one-shot health-check fallback to Anthropic.

    Order:
      1) FEDERAL_EAGLE_PROVIDER env override ("openai" or "anthropic").
      2) OpenAI if OPENAI_API_KEY is set and a tiny health check succeeds.
      3) Anthropic if ANTHROPIC_API_KEY is set.
    """
    forced = (os.getenv("FEDERAL_EAGLE_PROVIDER", "") or "").lower()
    if forced == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        return LLMChoice("anthropic", default_model_anthropic)
    if forced == "openai" and os.getenv("OPENAI_API_KEY"):
        return LLMChoice("openai", default_model_openai)

    if os.getenv("OPENAI_API_KEY") and _openai_healthy():
        return LLMChoice("openai", default_model_openai)

    if os.getenv("ANTHROPIC_API_KEY"):
        return LLMChoice(
            "anthropic", default_model_anthropic,
            fallback_used=True,
            fallback_reason="OpenAI health check failed or key missing",
        )

    # Last resort: try OpenAI anyway so the original error surfaces clearly.
    return LLMChoice("openai", default_model_openai,
                     fallback_used=False,
                     fallback_reason="No ANTHROPIC_API_KEY available for fallback")


def _openai_healthy(timeout_s: float = 4.0) -> bool:
    """Very light health-check. We don't want a true ping every request, just enough to fail fast."""
    cache = _OPENAI_HEALTH_CACHE
    now = time.time()
    if cache["expires_at"] > now:
        return cache["healthy"]
    healthy = False
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(timeout=timeout_s)
        # Lightweight: list models is cached server-side.
        client.models.list()
        healthy = True
    except Exception:
        healthy = False
    # Cache for 60s either way.
    cache["healthy"] = healthy
    cache["expires_at"] = now + 60
    return healthy


_OPENAI_HEALTH_CACHE: Dict[str, Any] = {"healthy": True, "expires_at": 0.0}


# ---------------------------------------------------------------------------
# Per-session rate limiter
# ---------------------------------------------------------------------------

class SessionRateLimiter:
    """Sliding-window counter keyed by session_id. Thread-safe.

    Defaults to 10 requests per 600 seconds. Override via env or constructor.
    """

    def __init__(self, max_calls: Optional[int] = None, window_s: Optional[float] = None):
        self.max_calls = max_calls or int(os.getenv("FEDERAL_EAGLE_MAX_CALLS_PER_SESSION", "10"))
        self.window_s = window_s or float(os.getenv("FEDERAL_EAGLE_RATE_WINDOW_S", "600"))
        self._lock = threading.Lock()
        self._calls: Dict[str, List[float]] = {}

    def check(self, session_id: str) -> Tuple[bool, int, int]:
        """Returns (allowed, remaining, retry_after_seconds_if_blocked)."""
        now = time.time()
        with self._lock:
            history = [t for t in self._calls.get(session_id, []) if (now - t) < self.window_s]
            if len(history) >= self.max_calls:
                oldest = history[0]
                retry_after = int(self.window_s - (now - oldest)) + 1
                self._calls[session_id] = history
                return False, 0, retry_after
            history.append(now)
            self._calls[session_id] = history
            return True, self.max_calls - len(history), 0


# ---------------------------------------------------------------------------
# Tavily cache (precedent stability)
# ---------------------------------------------------------------------------

class TavilyCache:
    """SQLite-backed cache for Tavily search results.

    Tavily's responses can vary call-to-call (ad-hoc ranking, freshness signals,
    region quirks). For the precedent agent we'd rather have stable outputs
    across runs even if that means slightly stale results, so we cache.
    """

    def __init__(self, path: Optional[str] = None, ttl_s: Optional[int] = None):
        default_path = Path(os.getenv("FEDERAL_EAGLE_TAVILY_CACHE", "logs/tavily_cache.sqlite"))
        default_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path or default_path)
        self.ttl_s = ttl_s if ttl_s is not None else int(os.getenv("FEDERAL_EAGLE_TAVILY_TTL_S", str(7 * 24 * 3600)))
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, ts REAL NOT NULL, value TEXT NOT NULL)"
            )

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path)
        try:
            yield c
            c.commit()
        finally:
            c.close()

    @staticmethod
    def _key(query: str, params: Optional[Dict] = None) -> str:
        return json.dumps({"q": (query or "").strip().lower(), "p": params or {}}, sort_keys=True)

    def get(self, query: str, params: Optional[Dict] = None) -> Optional[Any]:
        k = self._key(query, params)
        with self._lock, self._conn() as conn:
            row = conn.execute("SELECT ts, value FROM cache WHERE key = ?", (k,)).fetchone()
        if not row:
            return None
        ts, value = row
        if (time.time() - ts) > self.ttl_s:
            return None
        try:
            return json.loads(value)
        except Exception:
            return None

    def set(self, query: str, value: Any, params: Optional[Dict] = None) -> None:
        k = self._key(query, params)
        v = json.dumps(value, default=str)
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO cache(key, ts, value) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET ts=excluded.ts, value=excluded.value",
                (k, time.time(), v),
            )

    def memoize(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator that caches `fn(query, **params)`."""
        def wrapped(query: str, **params):
            hit = self.get(query, params)
            if hit is not None:
                return hit
            result = fn(query, **params)
            self.set(query, result, params)
            return result
        return wrapped


# Module-level singleton so all imports share the same cache file.
_GLOBAL_TAVILY_CACHE: Optional[TavilyCache] = None


def get_tavily_cache() -> TavilyCache:
    global _GLOBAL_TAVILY_CACHE
    if _GLOBAL_TAVILY_CACHE is None:
        _GLOBAL_TAVILY_CACHE = TavilyCache()
    return _GLOBAL_TAVILY_CACHE
