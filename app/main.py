"""MemoryAssist API - entry point. Run with:  uvicorn app.main:app --reload"""
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .database import create_indexes
from .routers import auth, interactions, items, recommendations


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_indexes()  # make sure all indexes exist when the server starts
    yield


app = FastAPI(
    title="MemoryAssist API",
    description=(
        "Backend of a memory recommendation platform for AR glasses. "
        "Users (glasses wearers) place everyday items, the glasses record gaze, "
        "and next-day recall shows which items were forgotten."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(auth.router)
app.include_router(items.router)
app.include_router(interactions.router)
app.include_router(recommendations.router)

# the web interface (Final Project): http://127.0.0.1:8000/app/
app.mount("/app", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="app")


@app.get("/", tags=["Health"])
def root():
    return {"service": "MemoryAssist API", "docs": "/docs", "app": "/app/"}
