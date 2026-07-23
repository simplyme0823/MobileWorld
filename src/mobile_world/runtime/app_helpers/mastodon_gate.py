"""Structured logging and error types for Mastodon initialization gates."""

import json
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

MASTODON_INIT_ERROR_CODE = "MOBILE_WORLD_MASTODON_INIT_FAILED"
MASTODON_GATE_LOG_PATH = Path(
    os.getenv(
        "MOBILE_WORLD_MASTODON_GATE_LOG",
        "/tmp/mobile-world-mastodon-gate.jsonl",
    )
)

_CURRENT_TASK_NAME: ContextVar[str | None] = ContextVar(
    "mastodon_initialization_task",
    default=None,
)
_LAST_INITIALIZATION_ERROR: ContextVar["MastodonInitializationError | None"] = ContextVar(
    "mastodon_initialization_error",
    default=None,
)


class MastodonInitializationError(RuntimeError):
    """Raised after all Mastodon backend initialization attempts fail."""

    def __init__(
        self,
        task_name: str,
        attempts: int,
        reason: str,
    ) -> None:
        super().__init__(
            f"[{MASTODON_INIT_ERROR_CODE}] Mastodon initialization failed "
            f"for {task_name} after {attempts} attempts: {reason}"
        )
        self.task_name = task_name
        self.attempts = attempts
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": MASTODON_INIT_ERROR_CODE,
            "task": self.task_name,
            "attempts": self.attempts,
            "reason": self.reason,
        }


@contextmanager
def initialization_context(task_name: str) -> Iterator[None]:
    """Associate gate records with the task currently being initialized."""

    task_token = _CURRENT_TASK_NAME.set(task_name)
    error_token = _LAST_INITIALIZATION_ERROR.set(None)
    try:
        yield
    finally:
        _CURRENT_TASK_NAME.reset(task_token)
        _LAST_INITIALIZATION_ERROR.reset(error_token)


def get_current_task_name() -> str:
    return _CURRENT_TASK_NAME.get() or "unknown"


def set_last_initialization_error(error: MastodonInitializationError) -> None:
    _LAST_INITIALIZATION_ERROR.set(error)


def get_last_initialization_error() -> MastodonInitializationError | None:
    return _LAST_INITIALIZATION_ERROR.get()


def record_gate_event(event: str, **fields: Any) -> None:
    """Write one JSONL event without allowing logging failures to break setup."""

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        "task": get_current_task_name(),
        "container": socket.gethostname(),
        **fields,
    }
    serialized = json.dumps(record, ensure_ascii=False, default=str)
    logger.info(f"[MASTODON_GATE] {serialized}")

    try:
        MASTODON_GATE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MASTODON_GATE_LOG_PATH.open("a", encoding="utf-8") as gate_log:
            gate_log.write(f"{serialized}\n")
    except Exception as error:
        logger.warning(f"Failed to write Mastodon gate log: {error}")
