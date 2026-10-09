import json
import os
from typing import Any, Dict

from evidence.evidence_store import get_event
from evidence.evidence_store import get_investigation


class ThreatIntelligenceAgent:

    def __init__(self):

        self.name = "Threat Intelligence Agent"
        self.version = "v1.0"

        kb_path = os.path.join(
            os.path.dirname(__file__),
            "threat_intel_kb.json"
        )

        with open(kb_path, "r", encoding="utf-8") as file:
            self.knowledge_base = json.load(file)

    def analyze(self, event_id: str) -> Dict[str, Any]:

        event = get_event(event_id)

        if event is None:
            raise ValueError(
                f"Evidence event not found: {event_id}"
            )

        investigation = get_investigation(event_id)

        attack_family = event.get("attack_family")

        if not attack_family:
            attack_family = "OTHER_ATTACK"

        intelligence = self.knowledge_base.get(
            attack_family,
            self.knowledge_base["OTHER_ATTACK"]
        )

        result = {
            "event_id": event_id,
            "incident_id": event.get("incident_id"),
            "attack_family": attack_family,
            "threat_category": intelligence["threat_category"],
            "risk_level": intelligence["risk_level"],
            "description": intelligence["description"],
            "mitre_attack": intelligence.get(
                "mitre_attack",
                {}
            ),
            "recommended_actions": intelligence[
                "recommended_actions"
            ],
            "detection_confidence": event.get(
                "confidence"
            ),
            "investigation_status": (
                investigation.get("status")
                if investigation
                else None
            ),
            "source_ip": event.get("source_ip"),
            "destination_ip": event.get("destination_ip"),
            "destination_port": event.get(
                "destination_port"
            ),
            "evidence_source": "CyberSecAI Local Threat Intelligence Knowledge Base",
            "agent": self.name,
            "agent_version": self.version
        }

        return result


if __name__ == "__main__":

    EVENT_ID = "EVT-0ECAC359F202"

    print("=" * 70)
    print("CYBERSECAI THREAT INTELLIGENCE AGENT")
    print("=" * 70)

    agent = ThreatIntelligenceAgent()

    result = agent.analyze(EVENT_ID)

    print("\nTHREAT INTELLIGENCE RESULT")
    print("=" * 70)

    for key, value in result.items():
        print(f"\n{key}: {value}")

    print("\n" + "=" * 70)
    print("THREAT INTELLIGENCE ANALYSIS COMPLETE")
    print("=" * 70)
