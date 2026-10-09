
import json
import sqlite3
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "data" / "evidence" / "evidence_store.db"


def _connect() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(DATABASE_PATH),
        timeout=30.0,
    )
    connection.row_factory = sqlite3.Row
    return connection


def initialize_report_table() -> None:
    """Create the report table without modifying existing evidence tables."""

    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_reports (
                report_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                incident_id TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                model TEXT NOT NULL,
                model_version TEXT NOT NULL,
                generation_method TEXT NOT NULL,
                report_json TEXT NOT NULL,
                report_version INTEGER NOT NULL,
                UNIQUE(event_id, report_version)
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_incident_reports_event_time
            ON incident_reports(event_id, generated_at DESC)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_incident_reports_incident
            ON incident_reports(incident_id)
            """
        )


def save_report(
    event_id: str,
    incident_id: str,
    model: str,
    model_version: str,
    generation_method: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Persist a report as an immutable, versioned JSON snapshot."""

    if not event_id or not incident_id:
        raise ValueError("event_id and incident_id are required.")

    if not isinstance(report, dict) or not report:
        raise ValueError("A non-empty report object is required.")

    # Validate JSON serializability before starting the transaction.
    report_json = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )

    report_id = f"RPT-{uuid.uuid4().hex[:12].upper()}"
    generated_at = datetime.now(timezone.utc).isoformat()

    initialize_report_table()

    # BEGIN IMMEDIATE serializes version allocation among writers.
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")

        row = connection.execute(
            """
            SELECT COALESCE(MAX(report_version), 0) + 1 AS next_version
            FROM incident_reports
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()

        report_version = int(row["next_version"])

        connection.execute(
            """
            INSERT INTO incident_reports (
                report_id,
                event_id,
                incident_id,
                generated_at,
                model,
                model_version,
                generation_method,
                report_json,
                report_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                event_id,
                incident_id,
                generated_at,
                model,
                model_version,
                generation_method,
                report_json,
                report_version,
            ),
        )

    return {
        "report_id": report_id,
        "event_id": event_id,
        "incident_id": incident_id,
        "generated_at": generated_at,
        "model": model,
        "model_version": model_version,
        "generation_method": generation_method,
        "report_version": report_version,
    }


def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
    try:
        report = json.loads(row["report_json"])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Stored report {row['report_id']} contains invalid JSON."
        ) from exc

    return {
        "report_id": row["report_id"],
        "event_id": row["event_id"],
        "incident_id": row["incident_id"],
        "generated_at": row["generated_at"],
        "model": row["model"],
        "model_version": row["model_version"],
        "generation_method": row["generation_method"],
        "report_version": row["report_version"],
        "report": report,
    }


def get_latest_report(event_id: str) -> dict[str, Any] | None:
    """Return the newest saved report for an event."""

    initialize_report_table()

    with _connect() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM incident_reports
            WHERE event_id = ?
            ORDER BY report_version DESC
            LIMIT 1
            """,
            (event_id,),
        ).fetchone()

    return _decode_row(row) if row else None


def get_report_by_id(report_id: str) -> dict[str, Any] | None:
    """Retrieve a specific report snapshot by report ID."""

    initialize_report_table()

    with _connect() as connection:
        row = connection.execute(
            """
            SELECT *
            FROM incident_reports
            WHERE report_id = ?
            LIMIT 1
            """,
            (report_id,),
        ).fetchone()

    return _decode_row(row) if row else None


def list_event_reports(
    event_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List report metadata and JSON snapshots for an event."""

    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")

    initialize_report_table()

    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM incident_reports
            WHERE event_id = ?
            ORDER BY report_version DESC
            LIMIT ?
            """,
            (event_id, limit),
        ).fetchall()

    return [_decode_row(row) for row in rows]
