import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter(
    prefix="/api/v1/decision",
    tags=["Decision Support"],
)

LAYA_SERVICE_URL = "http://127.0.0.1:8001"


@router.post("/{event_id}")
def make_decision(event_id: str):
    """Request an evidence-grounded decision from the local Laya service."""

    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(
                f"{LAYA_SERVICE_URL}/decision/{event_id}"
            )

        if response.status_code == 404:
            raise HTTPException(status_code=404, detail=response.text)

        if response.status_code == 422:
            raise HTTPException(status_code=422, detail=response.text)

        response.raise_for_status()
        return response.json()

    except HTTPException:
        raise
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Laya service is unavailable. Start the service on port 8001.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Laya service returned HTTP {exc.response.status_code}.",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not communicate with the Laya service: {exc}",
        ) from exc
