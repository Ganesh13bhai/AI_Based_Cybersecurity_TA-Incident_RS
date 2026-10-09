
from rag.document_ingestion import DocumentIngestion


documents = [
    {
        "title": "Network Service Scanning — MITRE ATT&CK T1046",
        "source": "MITRE ATT&CK Enterprise technique T1046",
        "text": (
            "Network Service Discovery involves adversaries identifying "
            "services running on remote hosts. Port scanning can be part "
            "of reconnaissance. A detector alert alone does not prove "
            "successful exploitation or host compromise. Investigators "
            "should examine connection logs, contacted ports, time ranges, "
            "and related events before drawing conclusions."
        ),
        "metadata": {
            "technique_id": "T1046",
            "tactic": "Discovery",
            "category": "network_reconnaissance",
        },
    },
    {
        "title": "Defensive triage for suspected port scans",
        "source": "CyberSecAI defensive knowledge base",
        "text": (
            "For a suspected port scan, verify the source and destination "
            "addresses, inspect available network telemetry, identify "
            "the ports contacted, search for repeated activity, and "
            "correlate events across the relevant time window. Record "
            "what is observed separately from hypotheses. Blocking a "
            "host or changing firewall rules requires human authorization."
        ),
        "metadata": {"category": "incident_response"},
    },
    {
        "title": "Interpreting machine-learning alerts",
        "source": "CyberSecAI model interpretation guidance",
        "text": (
            "A machine-learning prediction is a classification, not "
            "conclusive proof of malicious intent or compromise. Report "
            "the model, threshold, score, and available evidence. "
            "Confidence scores should not be described as calibrated "
            "probabilities unless calibration has been independently "
            "validated. Missing telemetry must be reported as an evidence "
            "gap, not as proof that an event did not occur."
        ),
        "metadata": {"category": "model_limitations"},
    },
]


def main():
    ingestion = DocumentIngestion()

    for item in documents:
        result = ingestion.ingest(
            text=item["text"],
            source=item["source"],
            title=item["title"],
            metadata=item["metadata"],
        )
        print("Added:", result["document_id"], result["title"])


if __name__ == "__main__":
    main()
