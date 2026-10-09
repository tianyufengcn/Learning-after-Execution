"""SQLite eval state for the two-stage workflow evaluator.
One task per (policy, sample); a worker
claims a sample and owns the full Stage0->render->score->Stage1->render->score
lifecycle. Atomic UPDATE claims with heartbeat leases, bounded stale requeue,
max attempts, and resume by skipping already-complete samples.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def _connect(db_path: Path, busy_timeout_ms: int = 30_000) -> sqlite3.Connection:
    import time as _time

    last_exc: Exception | None = None
    deadline = _time.time() + max(30.0, busy_timeout_ms / 1000.0)
    while True:
        try:
            conn = sqlite3.connect(
                str(db_path), timeout=max(1.0, busy_timeout_ms / 1000.0), isolation_level=None
            )
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or _time.time() >= deadline:
                raise
            last_exc = exc
            _time.sleep(1.0)
    raise RuntimeError(f"sqlite connect failed: {last_exc}")


class EvalState:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with _connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    policy TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    row_order INTEGER NOT NULL,
                    gt_image_path TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    worker TEXT, attempts INTEGER NOT NULL DEFAULT 0,
                    claimed_at REAL, heartbeat_at REAL, finished_at REAL,
                    error TEXT, result_path TEXT,
                    PRIMARY KEY (policy, sample_id)
                );
                CREATE INDEX IF NOT EXISTS idx_tasks ON tasks(policy, status, row_order);

                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    pid INTEGER,
                    gpu INTEGER,
                    policy TEXT,
                    status TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    current_sample TEXT
                );
                """
            )

    def init_tasks(self, policy: str, rows: list[dict[str, Any]]) -> int:
        inserted = 0
        with _connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            for order, row in enumerate(rows):
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO tasks(
                        policy, sample_id, row_order, gt_image_path
                    ) VALUES(?,?,?,?)
                    """,
                    (policy, row["sample_id"], order, row["gt_image_path"]),
                )
                inserted += int(cur.rowcount or 0)
            conn.execute("COMMIT")
        return inserted

    def _requeue_stale_locked(self, conn, policy: str, lease_seconds: float, max_attempts: int, now: float) -> None:
        cutoff = now - float(lease_seconds)
        stale = conn.execute(
            """
            SELECT sample_id, attempts FROM tasks
            WHERE policy=? AND status='running' AND COALESCE(heartbeat_at, claimed_at, 0) < ?
            """,
            (policy, cutoff),
        ).fetchall()
        for row in stale:
            if int(row["attempts"]) < int(max_attempts):
                conn.execute(
                    """
                    UPDATE tasks SET status='pending', worker=NULL, claimed_at=NULL,
                                    heartbeat_at=NULL WHERE policy=? AND sample_id=?
                    """,
                    (policy, row["sample_id"]),
                )
            else:
                conn.execute(
                    """
                    UPDATE tasks SET status='failed', error='stale lease at max attempts',
                                    finished_at=? WHERE policy=? AND sample_id=?
                    """,
                    (now, policy, row["sample_id"]),
                )

    def claim(self, policy: str, worker_id: str, lease_seconds: float = 600.0, max_attempts: int = 3) -> dict[str, Any] | None:
        now = time.time()
        with _connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._requeue_stale_locked(conn, policy, lease_seconds, max_attempts, now)
            row = conn.execute(
                """
                SELECT sample_id, gt_image_path FROM tasks
                WHERE policy=? AND status='pending' AND attempts < ?
                ORDER BY row_order LIMIT 1
                """,
                (policy, int(max_attempts)),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            conn.execute(
                """
                UPDATE tasks SET status='running', worker=?, attempts=attempts+1,
                                claimed_at=?, heartbeat_at=?
                WHERE policy=? AND sample_id=?
                """,
                (worker_id, now, now, policy, row["sample_id"]),
            )
            conn.execute("COMMIT")
        return {"sample_id": row["sample_id"], "gt_image_path": row["gt_image_path"]}

    def heartbeat(self, policy: str, sample_id: str, worker_id: str) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE tasks SET heartbeat_at=? WHERE policy=? AND sample_id=? AND worker=?
                """,
                (time.time(), policy, sample_id, worker_id),
            )

    def complete(self, policy: str, sample_id: str, worker_id: str, ok: bool, result_path: str, error: str | None) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE tasks SET status=?, finished_at=?, error=?, result_path=?
                WHERE policy=? AND sample_id=? AND worker=?
                """,
                ("done" if ok else "failed", time.time(), error, result_path, policy, sample_id, worker_id),
            )

    def reset_sample(self, policy: str, sample_id: str) -> int:
        with _connect(self.db_path) as conn:
            cur = conn.execute(
                """
                UPDATE tasks SET status='pending', worker=NULL, attempts=0,
                    claimed_at=NULL, heartbeat_at=NULL, finished_at=NULL,
                    error=NULL, result_path=NULL
                WHERE policy=? AND sample_id=?
                """,
                (policy, sample_id),
            )
            return int(cur.rowcount or 0)

    def register_worker(self, worker_id: str, role: str, pid: int, gpu: int, policy: str) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO workers(
                    worker_id, role, pid, gpu, policy, status, started_at, heartbeat_at
                ) VALUES(?,?,?,?,?,'alive',?,?)
                """,
                (worker_id, role, pid, gpu, policy, time.time(), time.time()),
            )

    def worker_heartbeat(self, worker_id: str, current_sample: str | None = None) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE workers SET heartbeat_at=?, current_sample=? WHERE worker_id=?
                """,
                (time.time(), current_sample, worker_id),
            )

    def stop_worker(self, worker_id: str) -> None:
        with _connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE workers SET status='stopped', heartbeat_at=? WHERE worker_id=?
                """,
                (time.time(), worker_id),
            )

    def counts(self, policy: str) -> dict[str, int]:
        with _connect(self.db_path) as conn:
            out = {"pending": 0, "running": 0, "done": 0, "failed": 0}
            for row in conn.execute(
                "SELECT status, COUNT(*) AS n FROM tasks WHERE policy=? GROUP BY status", (policy,)
            ):
                out[row["status"]] = int(row["n"])
            return out

    def list_workers(self) -> list[dict[str, Any]]:
        with _connect(self.db_path) as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM workers")]

    def worker_alive(self, worker_id: str, stale_s: float) -> bool:
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT heartbeat_at FROM workers WHERE worker_id=?", (worker_id,)).fetchone()
            return bool(row) and (time.time() - float(row["heartbeat_at"])) < float(stale_s)
