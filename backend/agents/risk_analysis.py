
"""
CyberSecAI Risk Analysis Agent

Produces explainable, rule-based risk assessments using stored
detection evidence, investigation findings, and correlation results.

Risk scores are heuristic scores, not probabilities of compromise.
No automatic remediation or incident merging is performed.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

from evidence.evidence_store import get_event, get_investigation


class RiskAnalysisAgent:
    def __init__(self):
        self.name = "Risk Analysis Agent"
        self.version = "v1.0"

    @staticmethod
    def _risk_level(score: int) -> str:
        if score >= 85:
            return "CRITICAL"
        if score >= 65:
            return "HIGH"
        if score >= 40:
            return "MEDIUM"
        return "LOW"

    def analyze(self, event_id: str) -> Dict[str, Any]:
        # 1. Load the original detection event
        event = get_event(event_id)

        if event is None:
            raise ValueError(f"Evidence event not found: {event_id}")

        investigation = get_investigation(event_id)

        prediction = str(
            event.get("prediction_label", "UNKNOWN")
        ).upper()

        confidence = float(event.get("confidence") or 0.0)
        confidence = max(0.0, min(1.0, confidence))

        attack_family = event.get("attack_family")
        incident_id = event.get("incident_id")

        score = 0
        factors: List[Dict[str, Any]] = []
        recommendations: List[str] = []

        def add_factor(name: str, points: int, reason: str):
            nonlocal score

            score += points
            factors.append({
                "factor": name,
                "points": points,
                "reason": reason,
            })

        # 2. Detection classification
        if prediction == "ATTACK":
            add_factor(
                "attack_detection",
                20,
                "The ML detector classified this event as ATTACK.",
            )
            recommendations.append(
                "Review the original network-flow evidence and detection."
            )
        elif prediction == "BENIGN":
            add_factor(
                "benign_detection",
                0,
                "The ML detector classified this event as BENIGN; "
                "this does not independently prove the activity is safe.",
            )
        else:
            add_factor(
                "unknown_detection",
                10,
                "The event has an unrecognized prediction label.",
            )
            recommendations.append(
                "Verify the detection label and source evidence."
            )

        # 3. Confidence contribution
        if prediction == "ATTACK":
            confidence_points = round(confidence * 25)

            add_factor(
                "detection_confidence",
                confidence_points,
                f"Recorded detection confidence is {confidence:.2%}. "
                "Confidence contributes to this heuristic score; "
                "it is not a calibrated risk probability.",
            )

        # 4. Attack-family contribution
        family = str(attack_family or "").upper()

        family_points = {
            "DOS": 15,
            "DDOS": 15,
            "BOT": 15,
            "BOTNET": 15,
            "INFILTRATION": 20,
            "WEB_ATTACK": 15,
            "BRUTE_FORCE": 10,
            "PORT_SCAN": 10,
            "PORTSCAN": 10,
        }.get(family, 5 if family else 0)

        if family:
            add_factor(
                "attack_family",
                family_points,
                f"Recorded attack family: {family}. "
                "The family contribution uses a predefined heuristic.",
            )

        # 5. Investigation severity
        severity = (
            str(investigation.get("severity", "UNKNOWN")).upper()
            if investigation
            else "UNKNOWN"
        )

        severity_points = {
            "LOW": 0,
            "MEDIUM": 10,
            "HIGH": 15,
            "CRITICAL": 20,
        }.get(severity, 0)

        if investigation:
            add_factor(
                "investigation_severity",
                severity_points,
                f"Investigation severity is {severity}.",
            )
        else:
            factors.append({
                "factor": "investigation_severity",
                "points": 0,
                "reason": "No saved investigation was found; "
                          "the assessment uses detection evidence only.",
            })

        # 6. Correlation context
        correlation = (
            investigation.get("correlation") or {}
            if investigation
            else {}
        )

        correlated_events = correlation.get("related_events", [])
        correlated_count = len(correlated_events)

        # Cap the contribution so a large number of matches cannot
        # dominate the entire assessment.
        correlation_points = min(correlated_count, 3) * 5

        add_factor(
            "incident_correlation",
            correlation_points,
            f"{correlated_count} potentially related event(s) were "
            "returned by the correlation agent. Correlation indicates "
            "similarity, not proof of a shared attack.",
        )

        if correlated_count:
            recommendations.append(
                "Review potentially related events and verify whether "
                "they belong to the same incident."
            )

        # 7. Bound the heuristic score
        score = max(0, min(100, score))
        risk_level = self._risk_level(score)

        # 8. Context-dependent recommendations
        if risk_level in ("HIGH", "CRITICAL"):
            recommendations.append(
                "Prioritize analyst review and follow the approved "
                "incident-response procedure."
            )
        elif risk_level == "MEDIUM":
            recommendations.append(
                "Review the evidence and monitor for additional "
                "suspicious activity."
            )
        else:
            recommendations.append(
                "Continue monitoring and reassess if new evidence appears."
            )

        if investigation:
            finding = investigation.get("finding")
        else:
            finding = None

        return {
            "event_id": event_id,
            "incident_id": incident_id,
            "risk_score": score,
            "risk_level": risk_level,
            "score_type": "heuristic_not_probability",
            "prediction_label": prediction,
            "attack_family": attack_family,
            "detection_confidence": confidence,
            "investigation_severity": severity,
            "correlated_event_count": correlated_count,
            "investigation_finding": finding,
            "factors": factors,
            "recommendations": recommendations,
            "assessment_method": "explainable_rule_based",
            "agent": self.name,
            "agent_version": self.version,
            "assessed_at": datetime.now(timezone.utc).isoformat(),
            "automatic_remediation_performed": False,
            "incident_merge_performed": False,
            "limitations": [
                "The score is heuristic and is not a probability of compromise.",
                "The weights are predefined and have not been calibrated "
                "against labeled risk outcomes.",
                "Missing investigation or correlation data can limit "
                "the assessment.",
                "Analyst verification is required before response actions.",
            ],
        }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python -m agents.risk_analysis <event_id>"
        )
        raise SystemExit(2)

    agent = RiskAnalysisAgent()

    try:
        result = agent.analyze(sys.argv[1])
        print(json.dumps(result, indent=2))
    except ValueError as exc:
        print(f"Risk analysis failed: {exc}")
        raise SystemExit(1)
