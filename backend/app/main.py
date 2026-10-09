from fastapi import FastAPI

from app.api.acquisition import router as acquisition_router
from app.api.profiler import router as profiler_router
from app.api.evidence import router as evidence_router
from app.api.investigation import router as investigation_router
from app.api.threat_intelligence import router as threat_intelligence_router
from app.api.decision import router as decision_router
from app.api.reports import router as reports_router

app = FastAPI(
    title="CyberSecAI",
    description="Evidence-Grounded Multi-Agent Cybersecurity Threat Analysis System",
    version="0.1.0"
)


# ============================================================
# API ROUTERS
# ============================================================

# Data acquisition and dataset discovery
app.include_router(acquisition_router)

# Dataset profiling
app.include_router(profiler_router)

# Evidence Store
app.include_router(evidence_router)

# Investigation Agent
app.include_router(investigation_router)

# Threat Intelligence Agent
app.include_router(threat_intelligence_router)

app.include_router(decision_router)

app.include_router(reports_router)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "system": "CyberSecAI",
        "status": "online",
        "mode": "offline",
        "version": "0.1.0"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "system": "CyberSecAI"
    }