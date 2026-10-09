"""
CyberSecAI Incident Correlation Agent

Identifies potentially related evidence events using explainable,
rule-based network and temporal signals.

This agent does not merge incidents or modify stored evidence.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from evidence.evidence_store import get_event, list_events


class IncidentCorrelationAgent:
    """Correlate suspicious events without automatically merging incidents."""

    def __init__(
        self,
        time_window_minutes: int = 60,
        minimum_score: float = 0.35,
        max_events: int = 500,
    ):
        if time_window_minutes <= 0:
            raise ValueError("time_window_minutes must be positive")
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between 0 and 1")
        if max_events < 1:
            raise ValueError("max_events must be positive")

        self.name = "Incident Correlation Agent"
        self.version = "v1.0"
        self.time_window_minutes = time_window_minutes
        self.minimum_score = minimum_score
        self.max_events = max_events

        self.weights = {
            "same_source_ip": 0.35,
            "same_destination_ip": 0.20,
            "same_destination_port": 0.10,
            "same_attack_family": 0.20,
            "time_proximity": 0.15,
        }

    @staticmethod
    def _parse_timestamp(value: Optional[str]):
        """Parse ISO timestamps, including timestamps ending in Z."""
        if not value:
            return None

        try:
            parsed = datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )

            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)

            return parsed
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _normalize(value):
        """Normalize non-empty string values for comparisons."""
        if value is None:
            return None

        value = str(value).strip()

        if not value:
            return None

        return value.casefold()

    def _compare(
        self,
        target: Dict[str, Any],
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare two events and explain their matching signals."""

        score = 0.0
        reasons: List[str] = []
        signals: List[str] = []

        comparisons = [
            (
                "source_ip",
                "same_source_ip",
                "Both events share the same source IP.",
            ),
            (
                "destination_ip",
                "same_destination_ip",
                "Both events share the same destination IP.",
            ),
            (
                "destination_port",
                "same_destination_port",
                "Both events target the same destination port.",
            ),
            (
                "attack_family",
                "same_attack_family",
                "Both events have the same attack family.",
            ),
        ]

        for field, signal, explanation in comparisons:
            left = self._normalize(target.get(field))
            right = self._normalize(candidate.get(field))

            # Missing values are not treated as matching evidence.
            if left is not None and right is not None and left == right:
                score += self.weights[signal]
                signals.append(signal)
                reasons.append(explanation)

        target_time = self._parse_timestamp(target.get("timestamp"))
        candidate_time = self._parse_timestamp(
            candidate.get("timestamp")
        )

        time_difference_minutes = None

        if target_time is not None and candidate_time is not None:
            time_difference_minutes = abs(
                (target_time - candidate_time).total_seconds()
            ) / 60.0

            if time_difference_minutes <= self.time_window_minutes:
                score += self.weights["time_proximity"]
                signals.append("time_proximity")
                reasons.append(
                    "Events occurred within the configured time window "
                    f"({time_difference_minutes:.1f} minutes apart)."
                )

        score = round(min(score, 1.0), 4)

        return {
            "event_id": candidate.get("event_id"),
            "incident_id": candidate.get("incident_id"),
            "correlation_score": score,
            "related": (
                score >= self.minimum_score
                and len(signals) >= 2
            ),
            "matching_signals": signals,
            "reasons": reasons,
            "time_difference_minutes": (
                round(time_difference_minutes, 2)
                if time_difference_minutes is not None
                else None
            ),
            "existing_incident_matches": (
                bool(target.get("incident_id"))
                and target.get("incident_id")
                == candidate.get("incident_id")
            ),
        }

    def correlate(self, event_id: str) -> Dict[str, Any]:
        """
        Find potentially related events.

        Reads existing events only. Does not create, merge, or update
        incidents or evidence records.
        """

        target = get_event(event_id)

        if target is None:
            raise ValueError(f"Evidence event not found: {event_id}")

        events = list_events(limit=self.max_events)
        comparisons = []
        examined = 0

        for candidate in events:
            candidate_id = candidate.get("event_id")

            if not candidate_id or candidate_id == event_id:
                continue

            examined += 1

            result = self._compare(target, candidate)

            if result["related"]:
                comparisons.append(result)

        comparisons.sort(
            key=lambda item: (
                item["correlation_score"],
                item["time_difference_minutes"] is not None,
            ),
            reverse=True,
        )

        return {
            "event_id": event_id,
            "incident_id": target.get("incident_id"),
            "status": "correlation_completed",
            "correlation_method": "explainable_rule_based",
            "agent": self.name,
            "agent_version": self.version,
            "time_window_minutes": self.time_window_minutes,
            "minimum_score": self.minimum_score,
            "events_examined": examined,
            "related_event_count": len(comparisons),
            "related_events": comparisons,
            "incident_merge_performed": False,
            "limitations": [
                "Correlation scores are rule-based similarity scores, "
                "not probabilities of an attack.",
                "Shared network attributes do not prove a common attacker.",
                "Only the most recent configured number of stored events "
                "are examined.",
                "Missing or invalid timestamps limit temporal comparison.",
                "Related events require analyst verification before "
                "incident merging or response actions.",
            ],
        }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python -m agents.incident_correlation EVENT_ID"
        )
        raise SystemExit(2)

    agent = IncidentCorrelationAgent()

    try:
        result = agent.correlate(sys.argv[1])
        print(json.dumps(result, indent=2))
    except ValueError as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)

