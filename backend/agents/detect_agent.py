"""
CyberSecAI Detection Agent

Connects ML detection results to the Evidence Store.
"""

from typing import Any, Dict, Optional

from evidence.evidence_store import create_event


class DetectAgent:

    def __init__(
        self,
        model_name: str = "Extra Trees",
        model_version: str = "cicids2017-v1",
        threshold: float = 0.05,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.threshold = threshold

    def process_detection(
        self,
        dataset: str,
        probability: float,
        features: Dict[str, Any],
        attack_family: Optional[str] = None,
        source_ip: Optional[str] = None,
        source_port: Optional[int] = None,
        destination_ip: Optional[str] = None,
        destination_port: Optional[int] = None,
        incident_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert an ML prediction into an evidence event.
        """

        prediction_label = (
            "ATTACK"
            if probability >= self.threshold
            else "BENIGN"
        )

        confidence = (
            probability
            if prediction_label == "ATTACK"
            else 1.0 - probability
        )

        evidence = []

        if prediction_label == "ATTACK":

            evidence.append(
                "ML model classified the network flow as ATTACK."
            )

            evidence.append(
                f"Attack probability "
                f"{probability:.4f} exceeds threshold "
                f"{self.threshold:.2f}."
            )

            if attack_family:
                evidence.append(
                    f"Predicted attack family: {attack_family}."
                )

        else:

            evidence.append(
                "ML model classified the network flow as BENIGN."
            )

        event_id = create_event(
            dataset=dataset,
            prediction_label=prediction_label,
            confidence=confidence,
            threshold=self.threshold,
            model_name=self.model_name,
            model_version=self.model_version,
            attack_family=attack_family,
            source_ip=source_ip,
            source_port=source_port,
            destination_ip=destination_ip,
            destination_port=destination_port,
            features=features,
            evidence=evidence,
            incident_id=incident_id,
            status="DETECTED",
        )

        return {
            "event_id": event_id,
            "prediction_label": prediction_label,
            "probability": probability,
            "confidence": confidence,
            "threshold": self.threshold,
            "attack_family": attack_family,
            "model": self.model_name,
            "status": "DETECTED",
        }
