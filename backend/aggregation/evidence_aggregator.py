from typing import Any, Dict

from evidence.evidence_store import get_event
from evidence.evidence_store import get_investigation

from threat_intelligence.threat_intelligence_agent import (
    ThreatIntelligenceAgent
)


class EvidenceAggregator:

    def __init__(self):

        self.name = "Evidence Aggregator"
        self.version = "v1.0"

        self.threat_intelligence_agent = (
            ThreatIntelligenceAgent()
        )

    def aggregate(
        self,
        event_id: str
    ) -> Dict[str, Any]:

        # ====================================================
        # 1. GET DETECTION EVIDENCE
        # ====================================================

        event = get_event(event_id)

        if event is None:
            raise ValueError(
                f"Evidence event not found: {event_id}"
            )

        # ====================================================
        # 2. GET INVESTIGATION
        # ====================================================

        investigation = get_investigation(event_id)

        if investigation is None:
            raise ValueError(
                f"Investigation not found for event: {event_id}"
            )

        # ====================================================
        # 3. GET THREAT INTELLIGENCE
        # ====================================================

        threat_intelligence = (
            self.threat_intelligence_agent.analyze(
                event_id
            )
        )

        # ====================================================
        # 4. BUILD DETECTION EVIDENCE
        # ====================================================

        detection_evidence = {
            "event_id": event.get("event_id"),
            "incident_id": event.get("incident_id"),
            "dataset": event.get("dataset"),
            "timestamp": event.get("timestamp"),
            "prediction_label": event.get(
                "prediction_label"
            ),
            "attack_family": event.get(
                "attack_family"
            ),
            "confidence": event.get(
                "confidence"
            ),
            "threshold": event.get(
                "threshold"
            ),
            "model_name": event.get(
                "model_name"
            ),
            "model_version": event.get(
                "model_version"
            ),
            "status": event.get(
                "status"
            ),
            "evidence": event.get(
                "evidence",
                []
            )
        }

        # ====================================================
        # 5. BUILD NETWORK EVIDENCE
        # ====================================================

        network_evidence = {
            "source_ip": event.get(
                "source_ip"
            ),
            "source_port": event.get(
                "source_port"
            ),
            "destination_ip": event.get(
                "destination_ip"
            ),
            "destination_port": event.get(
                "destination_port"
            )
        }

        # ====================================================
        # 6. BUILD INVESTIGATION EVIDENCE
        # ====================================================

        investigation_evidence = {
            "investigation_id": investigation.get(
                "investigation_id"
            ),
            "finding": investigation.get(
                "finding"
            ),
            "attack_family": investigation.get(
                "attack_family"
            ),
            "severity": investigation.get(
                "severity"
            ),
            "confidence": investigation.get(
                "confidence"
            ),
            "supporting_evidence": investigation.get(
                "supporting_evidence",
                []
            ),
            "related_events": investigation.get(
                "related_events",
                []
            ),
            "status": investigation.get(
                "status"
            )
        }

        # ====================================================
        # 7. BUILD THREAT INTELLIGENCE EVIDENCE
        # ====================================================

        threat_intelligence_evidence = {
            "threat_category": threat_intelligence.get(
                "threat_category"
            ),
            "risk_level": threat_intelligence.get(
                "risk_level"
            ),
            "description": threat_intelligence.get(
                "description"
            ),
            "mitre_attack": threat_intelligence.get(
                "mitre_attack",
                {}
            ),
            "recommended_actions": threat_intelligence.get(
                "recommended_actions",
                []
            ),
            "evidence_source": threat_intelligence.get(
                "evidence_source"
            ),
            "agent": threat_intelligence.get(
                "agent"
            ),
            "agent_version": threat_intelligence.get(
                "agent_version"
            )
        }

        # ====================================================
        # 8. CALCULATE OVERALL CONFIDENCE
        # ====================================================

        detection_confidence = float(
            event.get("confidence") or 0.0
        )

        investigation_confidence = float(
            investigation.get("confidence") or 0.0
        )

        overall_confidence = round(
            (
                detection_confidence
                + investigation_confidence
            ) / 2,
            4
        )

        # ====================================================
        # 9. BUILD FINAL AGGREGATED EVIDENCE
        # ====================================================

        aggregated_evidence = {

            "event_id": event.get(
                "event_id"
            ),

            "incident_id": event.get(
                "incident_id"
            ),

            "dataset": event.get(
                "dataset"
            ),

            "attack_family": event.get(
                "attack_family"
            ),

            "prediction": event.get(
                "prediction_label"
            ),

            "severity": investigation.get(
                "severity"
            ),

            "overall_confidence": overall_confidence,

            "detection": detection_evidence,

            "network": network_evidence,

            "investigation": investigation_evidence,

            "threat_intelligence":
                threat_intelligence_evidence,

            "evidence_chain": [
                "ML Detection",
                "Evidence Store",
                "Investigation",
                "Threat Intelligence"
            ],

            "ready_for_llm": True,

            "aggregator": self.name,

            "aggregator_version": self.version
        }

        return aggregated_evidence


if __name__ == "__main__":

    EVENT_ID = "EVT-0ECAC359F202"

    print("=" * 70)
    print("CYBERSECAI EVIDENCE AGGREGATOR")
    print("=" * 70)

    aggregator = EvidenceAggregator()

    result = aggregator.aggregate(
        EVENT_ID
    )

    print("\nAGGREGATED EVIDENCE")
    print("=" * 70)

    for key, value in result.items():
        print(f"\n{key}: {value}")

    print("\n" + "=" * 70)
    print("EVIDENCE AGGREGATION COMPLETE")
    print("=" * 70)
