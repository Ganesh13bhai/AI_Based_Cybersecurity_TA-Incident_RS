from fastapi import FastAPI, HTTPException

from aggregation.evidence_aggregator import EvidenceAggregator
from decision.laya_decision_agent import LayaDecisionAgent

app = FastAPI(title="CyberSecAI Laya Decision Service")

aggregator = EvidenceAggregator()
agent = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "laya-decision"}


@app.post("/decision/{event_id}")
def make_decision(event_id: str):
    global agent

    try:
        evidence = aggregator.aggregate(event_id)

        if not evidence.get("ready_for_llm"):
            raise HTTPException(
                status_code=422,
                detail="Evidence is not ready for decision analysis.",
            )

        if agent is None:
            agent = LayaDecisionAgent()

        decision = agent.decide(evidence)

        return {
            "status": "decision_generated",
            "event_id": event_id,
            "evidence": evidence,
            "decision": decision,
            "requires_human_review": True,
            "automated_remediation_authorized": False,
        }

    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Decision analysis failed: {exc}",
        ) from exc
