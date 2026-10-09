from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from evidence.evidence_store import get_investigation
from agents.investigate_agent import InvestigateAgent


router = APIRouter(
    prefix="/api/v1/investigation",
    tags=["Investigation"]
)


@router.post("/{event_id}")
def investigate_event(event_id: str) -> Dict[str, Any]:

    try:
        agent = InvestigateAgent()
        result = agent.investigate(event_id)

        return {
            "status": "success",
            "investigation": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


@router.get("/{event_id}")
def get_investigation_result(event_id: str) -> Dict[str, Any]:

    investigation = get_investigation(event_id)

    if investigation is None:
        raise HTTPException(
            status_code=404,
            detail=f"No investigation found for event: {event_id}"
        )

    return {
        "status": "success",
        "investigation": investigation
    }
