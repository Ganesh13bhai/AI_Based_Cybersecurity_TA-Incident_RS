from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from threat_intelligence.threat_intelligence_agent import (
    ThreatIntelligenceAgent
)


router = APIRouter(
    prefix="/api/v1/threat-intelligence",
    tags=["Threat Intelligence"]
)


@router.post("/{event_id}")
def analyze_threat(event_id: str) -> Dict[str, Any]:

    try:
        agent = ThreatIntelligenceAgent()
        result = agent.analyze(event_id)

        return {
            "status": "success",
            "threat_intelligence": result
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
def get_threat_intelligence(event_id: str) -> Dict[str, Any]:

    try:
        agent = ThreatIntelligenceAgent()
        result = agent.analyze(event_id)

        return {
            "status": "success",
            "threat_intelligence": result
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
