from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .project import ProjectStore


def _load(path: Path, default):
    return ProjectStore.load_json(path, default)


def _save(path: Path, data):
    ProjectStore.save_json(path, data)


def get_queue(store: ProjectStore) -> dict[str, Any]:
    return _load(store.queue_file, {"jobs": []}) or {"jobs": []}


def set_jobs(store: ProjectStore, jobs: list[dict[str, Any]]) -> None:
    normalized = []
    for job in jobs:
        item = dict(job)
        item.setdefault("id", str(uuid.uuid4()))
        item.setdefault("status", "waiting")
        item.setdefault("error", "")
        normalized.append(item)
    _save(store.queue_file, {"jobs": normalized})


def clear_finished(store: ProjectStore) -> None:
    data = get_queue(store)
    data["jobs"] = [j for j in data.get("jobs", []) if j.get("status") not in {"done", "failed"}]
    _save(store.queue_file, data)


def get_control(store: ProjectStore) -> dict[str, Any]:
    return _load(store.queue_control_file, {"paused": True, "stop_after_current": False}) or {}


def set_control(store: ProjectStore, **updates: Any) -> None:
    control = get_control(store)
    control.update(updates)
    _save(store.queue_control_file, control)


def worker_status(store: ProjectStore) -> dict[str, Any]:
    status = _load(store.worker_status_file, {}) or {}
    heartbeat = float(status.get("heartbeat", 0) or 0)
    status["active"] = bool(status.get("running") and time.time() - heartbeat < 12)
    return status


def start_worker(store: ProjectStore) -> int:
    status = worker_status(store)
    if status.get("active"):
        return int(status.get("pid", 0) or 0)

    script = Path(__file__).resolve().parents[2] / "scripts" / "queue_worker.py"
    args = [sys.executable, str(script), str(store.project_dir)]
    root = Path(__file__).resolve().parents[2]
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_log = log_dir / "queue_worker.out.log"
    err_log = log_dir / "queue_worker.err.log"

    kwargs: dict[str, Any] = {
        "cwd": str(root),
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True

    # Keep worker failures diagnosable. The child inherits duplicated handles;
    # the parent closes its handles immediately after Popen returns.
    with out_log.open("a", encoding="utf-8") as stdout_handle, err_log.open(
        "a",
        encoding="utf-8",
    ) as stderr_handle:
        process = subprocess.Popen(
            args,
            stdout=stdout_handle,
            stderr=stderr_handle,
            **kwargs,
        )

    return process.pid
