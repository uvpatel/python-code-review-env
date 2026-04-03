# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Code Review Environment.

This module creates an HTTP server that exposes the CodeReviewEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute a code review action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
    - POST /review/history: Submit code for review
    - GET /review/history: List previous reviews
    - DELETE /review/history: Clear review history
    - GET /review/config: Get reviewer configuration
    - PUT /review/config: Update reviewer configuration

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import (
        CodeReviewAction,
        CodeReviewConfig,
        CodeReviewObservation,
    )
    from code_review_environment import CodeReviewEnvironment
except ModuleNotFoundError:
    from models import (
        CodeReviewAction,
        CodeReviewConfig,
        CodeReviewObservation,
    )
    from code_review_environment import CodeReviewEnvironment

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class ReviewJob(BaseModel):
    """Stores a single code review job result."""

    job_id: str
    status: str = Field(default="completed")
    action: CodeReviewAction
    observation: CodeReviewObservation
    reward: float
    submitted_at: str
    completed_at: str
    logs: List[str] = Field(default_factory=list)


review_env = CodeReviewEnvironment()
REVIEW_STORE: Dict[str, ReviewJob] = {}
review_router = APIRouter(prefix="/review", tags=["review"])


@review_router.post("/history", response_model=ReviewJob)
def submit_review(action: CodeReviewAction):
    """Submit code for review and store the result."""
    observation = review_env.step(action)
    job_id = str(uuid4())
    now = datetime.utcnow().isoformat()
    job = ReviewJob(
        job_id=job_id,
        action=action,
        observation=observation,
        reward=observation.reward or 0.0,
        submitted_at=now,
        completed_at=now,
        logs=[observation.summary],
    )
    REVIEW_STORE[job_id] = job
    return job


@review_router.get("/history", response_model=List[ReviewJob])
def list_history(min_issues: Optional[int] = None):
    """List all review jobs, optionally filtered by minimum issue count."""
    entries = list(REVIEW_STORE.values())
    if min_issues is not None:
        entries = [job for job in entries if len(job.observation.issues) >= min_issues]
    return entries


@review_router.delete("/history")
def clear_history():
    """Clear all review history."""
    REVIEW_STORE.clear()
    review_env.clear_history()
    return JSONResponse({"detail": "history cleared"})


@review_router.put("/config", response_model=CodeReviewConfig)
def update_config(config: CodeReviewConfig):
    """Update the code reviewer configuration."""
    review_env.update_config(config)
    return review_env.config


@review_router.get("/config", response_model=CodeReviewConfig)
def get_config():
    """Get the current code reviewer configuration."""
    return review_env.config


# Create the OpenEnv app with code review environment
app = create_app(review_env)
app.include_router(review_router)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m python_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn python_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
