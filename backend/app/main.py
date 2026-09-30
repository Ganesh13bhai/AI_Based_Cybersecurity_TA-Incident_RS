from fastapi import FastAPI

from app.api.acquisition import router as acquisition_router
from app.api.profiler import router as profiler_router


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