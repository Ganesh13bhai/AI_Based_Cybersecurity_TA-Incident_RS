
import httpx

from fastapi import APIRouter, HTTPException, Query

from app.services.report_generator import IncidentReportGenerator
from reporting.report_store import (
    save_report,
    get_latest_report,
    get_report_by_id,
    list_event_reports,
)


router = APIRouter(
    prefix="/api/v1/reports",
    tags=["Incident Reports"],
)

LAYA_SERVICE_URL = "http://127.0.0.1:8001"
report_generator = IncidentReportGenerator()


@router.post("/{event_id}")
def generate_and_save_report(event_id: str):
    """Generate a report, persist it, and return the saved report metadata."""

    try:
        # Preserve the existing Laya integration.
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                f"{LAYA_SERVICE_URL}/decision/{event_id}"
            )

        if response.status_code == 404:
            raise HTTPException(
                status_code=404,
                detail="Event or investigation evidence was not found.",
            )

        if response.status_code == 422:
            raise HTTPException(
                status_code=422,
                detail=response.text,
            )

        response.raise_for_status()
        result = response.json()

        evidence = result.get("evidence")
        decision = result.get("decision")

        if not isinstance(evidence, dict) or not evidence:
            raise HTTPException(
                status_code=502,
                detail="Laya service returned no aggregated evidence.",
            )

        if not isinstance(decision, dict) or not decision:
            raise HTTPException(
                status_code=502,
                detail="Laya service returned no decision.",
            )

        if evidence.get("event_id") != event_id:
            raise HTTPException(
                status_code=502,
                detail="The returned evidence event ID does not match.",
            )

        generated = report_generator.generate(evidence, decision)

        report = generated.get("report")
        if not isinstance(report, dict) or not report:
            raise HTTPException(
                status_code=502,
                detail="Report generator returned no valid report object.",
            )

        incident_id = evidence.get("incident_id")
        if not incident_id:
            raise HTTPException(
                status_code=422,
                detail="Incident ID is missing from aggregated evidence.",
            )

        model = generated.get("model", "qwen3:8b")
        generation_method = generated.get(
            "generation_method",
            "qwen3",
        )

        metadata = save_report(
            event_id=event_id,
            incident_id=incident_id,
            model=model,
            model_version=model,
            generation_method=generation_method,
            report=report,
        )

        return {
            "status": "report_generated_and_saved",
            **metadata,
            "report": report,
            "limitations": generated.get("limitations", []),
        }

    except HTTPException:
        raise

    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Laya service is unavailable on port 8001.",
        ) from exc

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Timed out waiting for the Laya service.",
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Laya service returned HTTP {exc.response.status_code}.",
        ) from exc

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Report data could not be saved: {exc}",
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # Log the full exception in the backend terminal; don't expose
        # internal database paths or implementation details to clients.
        import logging
        logging.exception("Incident report generation/storage failed")

        raise HTTPException(
            status_code=500,
            detail="Incident report generation or storage failed.",
        ) from exc


@router.get("/{event_id}")
def retrieve_latest_report(event_id: str):
    """Retrieve the latest saved report without rerunning AI models."""

    try:
        saved = get_latest_report(event_id)

        if saved is None:
            raise HTTPException(
                status_code=404,
                detail="No saved report exists for this event.",
            )

        return {
            "status": "report_retrieved",
            **saved,
        }

    except HTTPException:
        raise

    except Exception as exc:
        import logging
        logging.exception("Saved report retrieval failed")

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve the saved report.",
        ) from exc


@router.get("/{event_id}/history")
def retrieve_report_history(
    event_id: str,
    limit: int = Query(default=20, ge=1, le=100),
):
    """Retrieve saved report versions for an event."""

    try:
        reports = list_event_reports(event_id, limit=limit)

        if not reports:
            raise HTTPException(
                status_code=404,
                detail="No saved reports exist for this event.",
            )

        return {
            "status": "report_history_retrieved",
            "event_id": event_id,
            "count": len(reports),
            "reports": reports,
        }

    except HTTPException:
        raise

    except Exception as exc:
        import logging
        logging.exception("Report history retrieval failed")

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve report history.",
        ) from exc


@router.get("/by-id/{report_id}")
def retrieve_report_by_id(report_id: str):
    """Retrieve a specific immutable report version."""

    try:
        saved = get_report_by_id(report_id)

        if saved is None:
            raise HTTPException(
                status_code=404,
                detail="Report ID was not found.",
            )

        return {
            "status": "report_retrieved",
            **saved,
        }

    except HTTPException:
        raise

    except Exception as exc:
        import logging
        logging.exception("Report-by-ID retrieval failed")

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve the requested report.",
        ) from exc
