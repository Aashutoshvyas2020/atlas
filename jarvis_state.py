from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_WAITING_CONFIRM = "waiting_confirmation"
STATUS_COMPLETED = "completed"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"


def now_ts() -> float:
    return time.time()


@dataclass
class TaskRecord:
    task_id: str
    goal: str
    status: str = STATUS_IDLE
    step: int = 0
    last_action: str = ""
    reason: str = ""
    result: str = ""
    created_at: float = field(default_factory=now_ts)
    updated_at: float = field(default_factory=now_ts)

    def touch(self) -> None:
        self.updated_at = now_ts()

    def to_payload(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "status": self.status,
            "step": self.step,
            "last_action": self.last_action,
            "reason": self.reason,
            "result": self.result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class PendingConfirmation:
    def __init__(self, task_id: str, summary: str):
        self.confirmation_id = f"confirm_{uuid.uuid4().hex[:10]}"
        self.task_id = task_id
        self.summary = summary
        self.created_at = now_ts()
        self._decision: Optional[str] = None
        self._event = threading.Event()

    def resolve(self, decision: str) -> None:
        val = (decision or "").strip().lower()
        if val not in {"approve", "reject"}:
            return
        self._decision = val
        self._event.set()

    def wait(self, timeout_s: float) -> Optional[str]:
        self._event.wait(max(0.1, float(timeout_s)))
        return self._decision


@dataclass
class SessionState:
    session_id: str
    active_task_id: str = ""
    pending_confirmation: Optional[PendingConfirmation] = None
