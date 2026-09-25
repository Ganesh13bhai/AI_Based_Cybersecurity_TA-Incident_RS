from fastapi import FastAPI

app = FastAPI(
    title="CyberSecAI",
    description="Evidence-Grounded Multi-Agent Cybersecurity Threat Analysis System",
    version="0.1.0"
)


@app.get("/")
def root():
    return {
        "system": "CyberSecAI",
        "status": "online",
        "mode": "offline"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }