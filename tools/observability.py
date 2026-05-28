"""Structured JSON logging for Federal Eagle.

One log file per run, written under logs/. Each line is a self-contained JSON
event with timestamp, run_id, stage, level, and an arbitrary payload. Designed
so you can grep/jq/ship to a log aggregator without parsing prose.

Usage:
    from tools.observability import get_logger, log_event, new_run

    run_id = new_run(label="streamlit_session_abc")
    log = get_logger(run_id)
    log("intake.start", payload={"user_input_len": len(user_input)})
    log("intake.done", payload={"duration_s": 1.2, "tokens_in": 420, "tokens_out": 180})

Event-name convention: '<stage>.<verb>' (e.g. 'crew.start', 'usc.tool_call',
'precedent.cache_hit', 'safety.block', 'drafter.repair', 'error').
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_LOG_DIR = Path(os.getenv("FEDERAL_EAGLE_LOG_DIR", "logs"))
_VERBOSE_TO_STDERR = os.getenv("FEDERAL_EAGLE_LOG_STDERR", "false").lower() == "true"


def _ensure_log_dir() -> Path:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR


def new_run(label: Optional[str] = None) -> str:
    """Allocate a new run_id and open the log file. Returns the run_id."""
    _ensure_log_dir()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
    path = _LOG_DIR / f"{run_id}.jsonl"
    # Open + write a header event so the file exists even if nothing else logs.
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "event": "run.start",
            "label": label,
            "pid": os.getpid(),
            "python": sys.version.split()[0],
        }) + "\n")
    return run_id


def get_logger(run_id: str) -> Callable[..., None]:
    """Return a `log(event, payload=None, level='info')` callable bound to run_id."""
    path = _LOG_DIR / f"{run_id}.jsonl"

    def log(event: str, payload: Optional[Dict[str, Any]] = None, level: str = "info") -> None:
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "event": event,
            "level": level,
            "payload": payload or {},
        }
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            # Logging must never crash the request.
            pass
        if _VERBOSE_TO_STDERR or level in ("error", "warn"):
            try:
                sys.stderr.write(json.dumps(rec, default=str) + "\n")
            except Exception:
                pass

    return log


# Convenience for code paths that don't have a run_id yet.
_global_run: Optional[str] = None


def _global_log_path() -> Path:
    global _global_run
    if _global_run is None:
        _global_run = new_run(label="global")
    return _LOG_DIR / f"{_global_run}.jsonl"


def log_event(event: str, payload: Optional[Dict[str, Any]] = None, level: str = "info") -> None:
    """One-shot logger that allocates a global run if needed."""
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": _global_run or "_pre_run",
        "event": event,
        "level": level,
        "payload": payload or {},
    }
    try:
        with _global_log_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


class StageTimer:
    """Tiny context manager that logs '<stage>.start' + '<stage>.done' with duration."""

    def __init__(self, log: Callable[..., None], stage: str, payload: Optional[Dict] = None):
        self._log = log
        self._stage = stage
        self._payload = payload or {}
        self._t0: Optional[float] = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._log(f"{self._stage}.start", payload=self._payload)
        return self

    def __exit__(self, exc_type, exc, tb):
        dur = (time.perf_counter() - self._t0) if self._t0 else 0.0
        if exc_type is not None:
            self._log(
                f"{self._stage}.error",
                payload={**self._payload, "duration_s": dur, "error_type": exc_type.__name__, "error": str(exc)},
                level="error",
            )
        else:
            self._log(f"{self._stage}.done", payload={**self._payload, "duration_s": dur})
