from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from evidence.evidence_store import (
    initialize_database,
    create_event,
    get_event,
    list_events,
    get_incident_events,
    update_event_status,
)


router = APIRouter(
    prefix="/api/v1/evidence",
    tags=["Evidence Store"]
)


class EvidenceCreate(BaseModel):
    dataset: str

    prediction_label: str

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )

    threshold: float = Field(
        ge=0.0,
        le=1.0
    )

    model_name: str

    attack_family: Optional[str] = None
    model_version: Optional[str] = None

    source_ip: Optional[str] = None
    source_port: Optional[int] = None

    destination_ip: Optional[str] = None
    destination_port: Optional[int] = None

    features: Dict[str, Any] = {}
    evidence: List[str] = []

    timestamp: Optional[str] = None
    incident_id: Optional[str] = None

    status: str = "NEW"


class StatusUpdate(BaseModel):
    status: str


@router.on_event("startup")
def startup():
    initialize_database()


@router.post("")
def create_evidence(data: EvidenceCreate):

    event_id = create_event(
        dataset=data.dataset,
        prediction_label=data.prediction_label,
        confidence=data.confidence,
        threshold=data.threshold,
        model_name=data.model_name,
        attack_family=data.attack_family,
        model_version=data.model_version,
        source_ip=data.source_ip,
        source_port=data.source_port,
        destination_ip=data.destination_ip,
        destination_port=data.destination_port,
        features=data.features,
        evidence=data.evidence,
        timestamp=data.timestamp,
        incident_id=data.incident_id,
        status=data.status,
    )

    return {
        "status": "success",
        "event_id": event_id,
    }


@router.get("")
def get_evidence(limit: int = 50):

    events = list_events(limit)

    return {
        "status": "success",
        "count": len(events),
        "events": events,
    }


@router.get("/incident/{incident_id}")
def get_incident_evidence(incident_id: str):

    events = get_incident_events(incident_id)

    return {
        "status": "success",
        "incident_id": incident_id,
        "count": len(events),
        "events": events,
    }


@router.get("/{event_id}")
def get_evidence_event(event_id: str):

    event = get_event(event_id)

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Evidence event not found"
        )

    return {
        "status": "success",
        "event": event,
    }


@router.patch("/{event_id}/status")
def update_status(
    event_id: str,
    data: StatusUpdate
):

    updated = update_event_status(
        event_id,
        data.status
    )

    if not updated:
        raise HTTPException(
            status_code=404,
            detail="Evidence event not found"
        )

    return {
        "status": "success",
        "event_id": event_id,
        "new_status": data.status,
    }
