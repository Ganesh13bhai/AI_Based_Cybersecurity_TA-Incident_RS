
from typing import Any, Dict, List

from evidence.evidence_store import (
    get_event,
    get_incident_events,
    save_investigation,
    update_event_status,
)

from agents.incident_correlation import IncidentCorrelationAgent


class InvestigateAgent:

    def __init__(self):
        self.name = "Investigate Agent"
        self.version = "v1.2"
        self.correlation_agent = IncidentCorrelationAgent()

    def investigate(self, event_id: str) -> Dict[str, Any]:
        # 1. Retrieve detection event
        event = get_event(event_id)

        if event is None:
            raise ValueError(
                f"Evidence event not found: {event_id}"
            )

        prediction = event["prediction_label"]
        confidence = float(event["confidence"])
        attack_family = event["attack_family"]
        incident_id = event["incident_id"]

        # 2. Determine preliminary severity
        if prediction == "BENIGN":
            severity = "LOW"
        elif confidence >= 0.90:
            severity = "HIGH"
        elif confidence >= 0.75:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # 3. Collect supporting evidence
        supporting_evidence: List[str] = []

        if prediction == "ATTACK":
            supporting_evidence.append(
                "ML detection classified the network flow as ATTACK."
            )

            threshold = float(event["threshold"])

            supporting_evidence.append(
                f"Detection confidence is {confidence:.2%}; "
                f"configured threshold is {threshold:.2%}."
            )

            if attack_family:
                supporting_evidence.append(
                    f"Detected attack family: {attack_family}."
                )

        elif prediction == "BENIGN":
            supporting_evidence.append(
                "ML detection classified the network flow as BENIGN. "
                "This classification does not independently prove "
                "that the activity is harmless."
            )

        if event.get("source_ip"):
            supporting_evidence.append(
                f"Source IP: {event['source_ip']}."
            )

        if event.get("source_port") is not None:
            supporting_evidence.append(
                f"Source port: {event['source_port']}."
            )

        if event.get("destination_ip"):
            supporting_evidence.append(
                f"Destination IP: {event['destination_ip']}."
            )

        if event.get("destination_port") is not None:
            supporting_evidence.append(
                f"Destination port: {event['destination_port']}."
            )

        for item in event.get("evidence", []):
            if item not in supporting_evidence:
                supporting_evidence.append(item)

        # 4. Generate investigation finding
        if prediction == "BENIGN":
            finding = (
                "The ML detector classified the event as BENIGN. "
                "No malicious activity was identified by this "
                "detection result."
            )

        elif attack_family == "PORT_SCAN":
            finding = (
                "Potential network scanning activity detected. "
                "The ML detector classified the event as PORT_SCAN. "
                "Analyst verification is required."
            )

        elif attack_family:
            finding = (
                f"Potential {attack_family} activity detected. "
                f"The ML detector classified the event as ATTACK "
                f"with {confidence:.2%} confidence."
            )

        else:
            finding = (
                "Potentially suspicious network activity detected. "
                "The ML detector classified the event as ATTACK."
            )

        # 5. Retrieve events already assigned to this incident
        related_events = []

        if incident_id:
            incident_events = get_incident_events(incident_id)

            for related in incident_events:
                if related["event_id"] == event_id:
                    continue

                related_events.append({
                    "event_id": related["event_id"],
                    "prediction_label": related["prediction_label"],
                    "attack_family": related["attack_family"],
                    "confidence": related["confidence"],
                    "status": related["status"],
                })

        # 6. Find potentially related events across incidents
        correlation_result = self.correlation_agent.correlate(
            event_id
        )

        correlated_events = correlation_result.get(
            "related_events", []
        )

        # Add correlation explanations to supporting evidence
        for correlated in correlated_events:
            score = float(
                correlated.get("correlation_score", 0.0)
            )

            matching_signals = correlated.get(
                "matching_signals", []
            )

            evidence_item = (
                f"Potentially related event "
                f"{correlated['event_id']}; "
                f"correlation score: {score:.2f}. "
                f"Matching signals: "
                f"{', '.join(matching_signals)}."
            )

            if evidence_item not in supporting_evidence:
                supporting_evidence.append(evidence_item)

        # 7. Persist investigation and full correlation result.
        # Correlation is evidence of a possible relationship, not
        # proof that events belong to the same incident.
        # No automatic incident merging is performed.
        investigation_id = save_investigation(
            event_id=event_id,
            incident_id=incident_id,
            finding=finding,
            attack_family=attack_family,
            severity=severity,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            related_events=related_events,
            correlation_result=correlation_result,
        )

        # 8. Update event status
        update_event_status(
            event_id,
            "INVESTIGATED",
        )

        # 9. Return complete investigation result
        return {
            "investigation_id": investigation_id,
            "event_id": event_id,
            "incident_id": incident_id,
            "finding": finding,
            "attack_family": attack_family,
            "severity": severity,
            "confidence": confidence,
            "supporting_evidence": supporting_evidence,

            # Existing same-incident relationships
            "related_events": related_events,
            "related_event_count": len(related_events),

            # Rule-based potential relationships
            "correlated_events": correlated_events,
            "correlated_event_count": len(correlated_events),
            "correlation_method": correlation_result.get(
                "correlation_method"
            ),
            "correlation_limitations": correlation_result.get(
                "limitations", []
            ),

            # Full correlation metadata is persisted in SQLite.
            "correlation": correlation_result,

            "incident_merge_performed": False,
            "status": "INVESTIGATED",
            "agent": self.name,
            "agent_version": self.version,
        }


if __name__ == "__main__":
    import json

    EVENT_ID = "EVT-0ECAC359F202"

    print("=" * 70)
    print("CYBERSECAI INVESTIGATE AGENT")
    print("=" * 70)

    agent = InvestigateAgent()

    try:
        result = agent.investigate(EVENT_ID)
        print(json.dumps(result, indent=2))

    except (ValueError, KeyError) as exc:
        print(f"Investigation failed: {exc}")
        raise SystemExit(1)

    print("=" * 70)
    print("INVESTIGATION COMPLETE")
    print("=" * 70)
