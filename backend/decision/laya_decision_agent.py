
"""Laya-based structured decision agent for CyberSecAI.

Run tests with:
    ~/mac-ml/bin/python -m decision.laya_decision_agent
"""

from __future__ import annotations

import json
from typing import Any

import laya


class LayaDecisionAgent:
    """Produce bounded decisions from existing cybersecurity evidence."""

    def __init__(self) -> None:
        # Load the already-downloaded typed-decisions checkpoint.
        self.model = laya.load(
            "convaiinnovations/laya",
            subfolder="typed-decisions",
        )

    @staticmethod
    def _questions() -> dict[str, dict[str, Any]]:
        return {
            "escalate": {
                "type": "noul",
                "instructions": (
                    "Does the supplied evidence justify escalating this "
                    "cybersecurity event for human security-team review?"
                ),
            },
            "severity": {
                "type": "choice",
                "instructions": (
                    "Based only on the supplied incident evidence, "
                    "which provisional severity is most appropriate?"
                ),
                "criteria": {
                    "low": "Low impact or weak evidence of malicious activity.",
                    "medium": "Suspicious activity requiring investigation.",
                    "high": "Strong evidence of a potentially significant attack.",
                    "critical": (
                        "Evidence of an active, severe incident with "
                        "potentially major impact."
                    ),
                },
            },
            "response": {
                "type": "choice",
                "instructions": (
                    "Which next response step is most appropriate, "
                    "considering the supplied evidence?"
                ),
                "criteria": {
                    "monitor": "Continue monitoring with no immediate escalation.",
                    "investigate": "Collect and review additional evidence.",
                    "contain": "Recommend containment for authorized human review.",
                    "escalate": "Escalate to the security team for review.",
                },
            },
            "evidence_sufficient": {
                "type": "noul",
                "instructions": (
                    "Is the supplied evidence sufficient to support "
                    "the stated provisional security assessment?"
                ),
            },
        }

    def decide(self, evidence: dict[str, Any]) -> dict[str, Any]:
        """Return Laya's raw typed answers and auditable decision metadata."""
        if not isinstance(evidence, dict) or not evidence:
            raise ValueError("A non-empty evidence dictionary is required.")

        state = {
            "task": (
                "CyberSecAI decision support. Assess the event using "
                "only the supplied evidence. Do not invent missing facts."
            ),
            "incident_evidence": evidence,
        }

        result = self.model.predict(state, self._questions())

        # Keep raw answers for auditing; do not treat model outputs as
        # authorization to execute remediation.
        return {
            "agent": "LayaDecisionAgent",
            "model": result.get("model", "laya-typed-decisions"),
            "event_id": evidence.get("event_id"),
            "incident_id": evidence.get("incident_id"),
            "answers": result.get("answers", {}),
            "routing": result.get("routing", {}),
            "usage": result.get("usage", {}),
            "confidence_calibration_warning": (
                "The checkpoint emitted a temperature warning during the "
                "previous CLI test. Do not treat its confidence values as "
                "calibrated until this checkpoint is independently validated."
            ),
            "automated_remediation_authorized": False,
        }



def main() -> None:
    """Test Laya using real evidence from CyberSecAI's SQLite store."""
    import json

    from aggregation.evidence_aggregator import EvidenceAggregator

    event_id = "EVT-0ECAC359F202"

    # Retrieve and combine real detection, investigation,
    # network, and threat-intelligence evidence.
    aggregator = EvidenceAggregator()
    aggregated_evidence = aggregator.aggregate(event_id)

    print("\n=== AGGREGATED EVIDENCE ===")
    print(json.dumps(aggregated_evidence, indent=2, default=str))

    if not aggregated_evidence.get("ready_for_llm"):
        raise RuntimeError("Evidence is not ready for decision analysis.")

    # Ask Laya to assess the actual aggregated evidence.
    agent = LayaDecisionAgent()
    decision = agent.decide(aggregated_evidence)

    print("\n=== LAYA DECISION ===")
    print(json.dumps(decision, indent=2, default=str))


if __name__ == "__main__":
    main()
