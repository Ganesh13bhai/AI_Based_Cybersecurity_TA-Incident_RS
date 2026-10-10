
"""
CyberSecAI Evidence Store

Local SQLite-backed evidence storage for detected cybersecurity events
and investigation results, including persistent incident correlation data.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DB_DIR = ROOT / "data" / "evidence"
DB_PATH = DB_DIR / "evidence_store.db"


def get_connection():
    DB_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    return conn


def initialize_database():
    """Create the Evidence Store database and tables."""

    conn = get_connection()

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evidence_events (
                event_id TEXT PRIMARY KEY,
                incident_id TEXT,
                dataset TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source_ip TEXT,
                source_port INTEGER,
                destination_ip TEXT,
                destination_port INTEGER,
                prediction_label TEXT NOT NULL,
                attack_family TEXT,
                confidence REAL NOT NULL,
                threshold REAL NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT,
                features_json TEXT,
                evidence_json TEXT,
                status TEXT NOT NULL DEFAULT 'NEW',
                created_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_evidence_incident
            ON evidence_events(incident_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_evidence_dataset
            ON evidence_events(dataset)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_evidence_status
            ON evidence_events(status)
        """)

        conn.commit()
    finally:
        conn.close()


def create_event(
    dataset,
    prediction_label,
    confidence,
    threshold,
    model_name,
    attack_family=None,
    model_version=None,
    source_ip=None,
    source_port=None,
    destination_ip=None,
    destination_port=None,
    features=None,
    evidence=None,
    timestamp=None,
    incident_id=None,
    status="NEW",
):
    """Store one ML detection event as evidence."""

    event_id = f"EVT-{uuid.uuid4().hex[:12].upper()}"

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    created_at = datetime.now(timezone.utc).isoformat()

    conn = get_connection()

    try:
        conn.execute(
            """
            INSERT INTO evidence_events (
                event_id,
                incident_id,
                dataset,
                timestamp,
                source_ip,
                source_port,
                destination_ip,
                destination_port,
                prediction_label,
                attack_family,
                confidence,
                threshold,
                model_name,
                model_version,
                features_json,
                evidence_json,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                incident_id,
                dataset,
                timestamp,
                source_ip,
                source_port,
                destination_ip,
                destination_port,
                prediction_label,
                attack_family,
                float(confidence),
                float(threshold),
                model_name,
                model_version,
                json.dumps(features or {}),
                json.dumps(evidence or []),
                status,
                created_at,
            ),
        )

        conn.commit()
    finally:
        conn.close()

    return event_id


def get_event(event_id):
    """Retrieve one evidence event."""

    conn = get_connection()

    try:
        row = conn.execute(
            """
            SELECT *
            FROM evidence_events
            WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    return row_to_dict(row)


def list_events(limit=50):
    """Retrieve recent evidence events."""

    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT *
            FROM evidence_events
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    return [row_to_dict(row) for row in rows]


def get_incident_events(incident_id):
    """Retrieve all evidence belonging to an incident."""

    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT *
            FROM evidence_events
            WHERE incident_id = ?
            ORDER BY created_at ASC
            """,
            (incident_id,),
        ).fetchall()
    finally:
        conn.close()

    return [row_to_dict(row) for row in rows]


def update_event_status(event_id, status):
    """Update the status of an evidence event."""

    conn = get_connection()

    try:
        cursor = conn.execute(
            """
            UPDATE evidence_events
            SET status = ?
            WHERE event_id = ?
            """,
            (status, event_id),
        )

        conn.commit()

        return cursor.rowcount > 0
    finally:
        conn.close()


def row_to_dict(row):
    """Convert an evidence database row into a dictionary."""

    data = dict(row)

    data["features"] = json.loads(
        data.pop("features_json") or "{}"
    )

    data["evidence"] = json.loads(
        data.pop("evidence_json") or "[]"
    )

    return data


def initialize_investigation_table():
    """
    Create investigation storage and safely migrate existing databases.

    Existing investigation records are preserved when correlation_json
    is added to an older database.
    """

    conn = get_connection()

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS investigations (
                investigation_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                incident_id TEXT,
                finding TEXT NOT NULL,
                attack_family TEXT,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                supporting_evidence_json TEXT,
                related_events_json TEXT,
                status TEXT NOT NULL DEFAULT 'INVESTIGATED',
                created_at TEXT NOT NULL,
                correlation_json TEXT
            )
        """)

        # Migration for databases created before correlation persistence.
        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(investigations)"
            ).fetchall()
        }

        if "correlation_json" not in columns:
            conn.execute("""
                ALTER TABLE investigations
                ADD COLUMN correlation_json TEXT
            """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_investigation_event
            ON investigations(event_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_investigation_incident
            ON investigations(incident_id)
        """)

        conn.commit()
    finally:
        conn.close()


def save_investigation(
    event_id,
    incident_id,
    finding,
    attack_family,
    severity,
    confidence,
    supporting_evidence,
    related_events,
    status="INVESTIGATED",
    correlation_result=None,
):
    """
    Persist an investigation and its optional correlation result.

    correlation_result is optional to preserve compatibility with
    existing callers that do not provide correlation information.
    """

    initialize_investigation_table()

    investigation_id = (
        f"INV-{uuid.uuid4().hex[:12].upper()}"
    )

    created_at = datetime.now(timezone.utc).isoformat()

    correlation_json = (
        json.dumps(correlation_result)
        if correlation_result is not None
        else None
    )

    conn = get_connection()

    try:
        conn.execute(
            """
            INSERT INTO investigations (
                investigation_id,
                event_id,
                incident_id,
                finding,
                attack_family,
                severity,
                confidence,
                supporting_evidence_json,
                related_events_json,
                status,
                created_at,
                correlation_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                investigation_id,
                event_id,
                incident_id,
                finding,
                attack_family,
                severity,
                float(confidence),
                json.dumps(supporting_evidence),
                json.dumps(related_events),
                status,
                created_at,
                correlation_json,
            ),
        )

        conn.commit()
    finally:
        conn.close()

    return investigation_id


def get_investigation(event_id):
    """
    Retrieve the latest investigation for an event, including its
    persisted correlation result.

    Older records without correlation data return correlation=None.
    """

    initialize_investigation_table()

    conn = get_connection()

    try:
        row = conn.execute(
            """
            SELECT *
            FROM investigations
            WHERE event_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (event_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    data = dict(row)

    data["supporting_evidence"] = json.loads(
        data.pop("supporting_evidence_json") or "[]"
    )

    data["related_events"] = json.loads(
        data.pop("related_events_json") or "[]"
    )

    correlation_json = data.pop("correlation_json", None)

    data["correlation"] = (
        json.loads(correlation_json)
        if correlation_json
        else None
    )

    return data


if __name__ == "__main__":
    print("=" * 70)
    print("CyberSecAI EVIDENCE STORE")
    print("=" * 70)

    initialize_database()
    initialize_investigation_table()

    print("\nDatabase initialized:")
    print(DB_PATH)

    event_id = create_event(
        dataset="CIC-IDS2017",
        prediction_label="ATTACK",
        confidence=0.97,
        threshold=0.05,
        model_name="Extra Trees",
        model_version="cicids2017-v1",
        attack_family="PORT_SCAN",
        source_ip="192.168.1.100",
        source_port=49152,
        destination_ip="192.168.1.10",
        destination_port=80,
        features={
            "Destination Port": 80,
            "Flow Duration": 120,
            "Total Fwd Packets": 15,
            "Total Backward Packets": 3,
            "Fwd_Bwd_Packet_Ratio": 5.0,
        },
        evidence=[
            "High connection frequency",
            "Repeated destination probing",
            "ML confidence above detection threshold",
        ],
        incident_id="INC-000001",
    )

    print("\nTest event created:")
    print(event_id)

    event = get_event(event_id)

    print("\nRetrieved event:")
    print(json.dumps(event, indent=2))

    print("\nUpdating status...")

    update_event_status(event_id, "DETECTED")

    event = get_event(event_id)

    print("Status:", event["status"])
    print("\nEvidence Store test: PASS")
    print("=" * 70)
