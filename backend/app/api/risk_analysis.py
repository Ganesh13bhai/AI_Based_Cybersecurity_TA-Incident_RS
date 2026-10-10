
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from agents.risk_analysis import RiskAnalysisAgent


router = APIRouter(
    prefix="/api/v1/risk-analysis",
    tags=["Risk Analysis"],
)


@router.post("/{event_id}")
def analyze_event_risk(event_id: str) -> Dict[str, Any]:
    """Generate an explainable risk assessment for an evidence event."""

    try:
        agent = RiskAnalysisAgent()
        result = agent.analyze(event_id)

        return {
            "status": "success",
            "risk_assessment": result,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Risk analysis failed.",
        ) from error
