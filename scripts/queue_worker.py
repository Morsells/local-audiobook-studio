from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audiobook_studio.privacy import activate_privacy_lock
activate_privacy_lock()

from src.audiobook_studio.models import BuildOptions, OutputMode
from src.audiobook_studio.pipeline import build_audio
from src.audiobook_studio.project import ProjectStore


def load(path: Path, default):
    return ProjectStore.load_json(path, default)


def save(path: Path, data):
    ProjectStore.save_json(path, data)


def status(store: ProjectStore, **updates):
    data = load(store.worker_status_file, {}) or {}
    data.update(updates)
    data["heartbeat"] = time.time()
    save(store.worker_status_file, data)


def main(project_dir: str) -> int:
    store = ProjectStore.from_existing_dir(project_dir)
    status(store, running=True, pid=os.getpid(), message="Worker started")
    try:
        while True:
            control = load(store.queue_control_file, {"paused": True, "stop_after_current": False}) or {}
            if control.get("paused", True):
                status(store, running=True, message="Paused")
                time.sleep(1.0)
                continue

            queue = load(store.queue_file, {"jobs": []}) or {"jobs": []}
            jobs = queue.get("jobs", [])
            next_index = next((i for i, j in enumerate(jobs) if j.get("status") in {"waiting", "retry"}), None)
            if next_index is None:
                status(store, running=False, message="Queue complete")
                return 0

            job = jobs[next_index]
            job["status"] = "running"
            job["started_at"] = datetime.now(timezone.utc).isoformat()
            queue["jobs"] = jobs
            save(store.queue_file, queue)
            status(store, running=True, current_job=job.get("id"), message=f"Generating {job.get('title', 'job')}")

            try:
                _, _, sections = store.load_analysis()
                selected_ids = set(job.get("section_ids", []))
                selected = [s for s in sections if s.id in selected_ids]
                if not selected:
                    raise RuntimeError("Queued section IDs are no longer present in saved analysis.")

                options = BuildOptions.from_dict(job["options"])
                options.output_mode = OutputMode.SINGLE
                overrides = store.load_overrides()

                def progress(message: str, fraction: float):
                    status(
                        store,
                        running=True,
                        current_job=job.get("id"),
                        message=message,
                        progress=float(fraction),
                    )

                outputs = build_audio(
                    store.project_dir,
                    selected,
                    options,
                    overrides=overrides,
                    progress=progress,
                    output_basename=job.get("output_name") or job.get("title") or "chapter",
                )
                queue = load(store.queue_file, {"jobs": []}) or {"jobs": []}
                jobs = queue.get("jobs", [])
                for item in jobs:
                    if item.get("id") == job.get("id"):
                        item["status"] = "done"
                        item["outputs"] = [str(p.name) for p in outputs]
                        item["finished_at"] = datetime.now(timezone.utc).isoformat()
                        item["error"] = ""
                save(store.queue_file, {"jobs": jobs})
            except Exception as exc:
                queue = load(store.queue_file, {"jobs": []}) or {"jobs": []}
                jobs = queue.get("jobs", [])
                for item in jobs:
                    if item.get("id") == job.get("id"):
                        item["status"] = "failed"
                        item["error"] = f"{type(exc).__name__}: {exc}"
                save(store.queue_file, {"jobs": jobs})
                status(store, running=True, message=f"Failed: {exc}")

            control = load(store.queue_control_file, {}) or {}
            if control.get("stop_after_current"):
                control["paused"] = True
                control["stop_after_current"] = False
                save(store.queue_control_file, control)

    finally:
        try:
            status(store, running=False, current_job=None)
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: queue_worker.py PROJECT_DIR")
    raise SystemExit(main(sys.argv[1]))
