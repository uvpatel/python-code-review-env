"""FastAPI app for the Python code-review environment.

This module exposes two layers of API surface:

- the OpenEnv-compatible routes created by `create_app(...)`
- additional convenience routes for health, tasks, grading, and history

The OpenEnv routes are what automated validators and agent clients care about.
The extra REST routes are there to make manual testing and debugging easier.
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required for the server. Install project dependencies before running."
    ) from exc

try:
    from models import (
        DeleteResponse,
        DirectReviewRequest,
        DirectReviewResponse,
        EpisodeRecord,
        HealthResponse,
        PythonEnvConfig,
        PythonReviewAction,
        PythonReviewObservation,
        TaskDescriptor,
        TaskEvaluation,
    )
    from server.code_review_environment import PythonEnvironment
except ModuleNotFoundError:  # pragma: no cover
    from ..models import (
        DeleteResponse,
        DirectReviewRequest,
        DirectReviewResponse,
        EpisodeRecord,
        HealthResponse,
        PythonEnvConfig,
        PythonReviewAction,
        PythonReviewObservation,
        TaskDescriptor,
        TaskEvaluation,
    )
    from .code_review_environment import PythonEnvironment

# Read deployment-related settings from environment variables so local Docker,
# HF Spaces, and larger deployments can all reuse the same application code.
MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "32"))

# The custom REST routes below use a singleton environment instance for simple
# debugging and manual inspection.
python_env = PythonEnvironment()

# The installed OpenEnv server expects a class or factory function here rather
# than a pre-instantiated environment object, so we pass `PythonEnvironment`.
app = create_app(
    PythonEnvironment,
    PythonReviewAction,
    PythonReviewObservation,
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)
router = APIRouter(tags=["python-env"])


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the root path to the OpenAPI docs for easier manual testing."""

    return RedirectResponse(url="/docs")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a lightweight health payload for local checks and deployments."""

    # Tasks are static, so this is a cheap way to confirm the app loaded the
    # benchmark definition correctly.
    tasks = python_env.list_tasks()
    active_episode_id = None
    active_task_id = tasks[0].task_id if tasks else None

    # If the singleton environment has processed any requests, surface the most
    # recent episode/task identifiers for easier debugging.
    if python_env.get_history():
        active = python_env.get_history()[-1]
        active_episode_id = active.episode_id
        active_task_id = active.task_id
    return HealthResponse(
        task_count=len(tasks),
        active_task_id=active_task_id,
        active_episode_id=active_episode_id,
    )


@router.get("/tasks", response_model=list[TaskDescriptor])
def list_tasks() -> list[TaskDescriptor]:
    """Return the public metadata for every benchmark task."""

    return python_env.list_tasks()


@router.get("/tasks/{task_id}", response_model=TaskDescriptor)
def get_task(task_id: str) -> TaskDescriptor:
    """Return one task descriptor by id.

    Args:
        task_id: Stable task identifier such as ``py-review-easy``.
    """

    try:
        return python_env.get_task(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/tasks/{task_id}/grade", response_model=TaskEvaluation)
def grade_task(task_id: str, payload: PythonReviewAction) -> TaskEvaluation:
    """Grade a proposed submission without stepping through an episode."""

    try:
        return python_env.grade_task_submission(
            task_id=task_id,
            findings=payload.findings,
            patched_code=payload.patched_code,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/review", response_model=DirectReviewResponse)
def direct_review(request: DirectReviewRequest) -> DirectReviewResponse:
    """Run the direct static-review helper on arbitrary Python code."""

    return python_env.direct_review(code=request.code, context=request.context)


@router.get("/history", response_model=list[EpisodeRecord])
def get_history() -> list[EpisodeRecord]:
    """Return the singleton environment's stored episode summaries."""

    return python_env.get_history()


@router.delete("/history", response_model=DeleteResponse)
def clear_history() -> DeleteResponse:
    """Clear the singleton environment's stored history."""

    python_env.clear_history()
    return DeleteResponse(detail="history cleared")


@router.get("/config", response_model=PythonEnvConfig)
def get_config() -> PythonEnvConfig:
    """Expose the current configuration used by the singleton environment."""

    return python_env.config


@router.put("/config", response_model=PythonEnvConfig)
def update_config(config: PythonEnvConfig) -> PythonEnvConfig:
    """Replace the singleton environment config with a new config payload."""

    python_env.update_config(config)
    return python_env.config


# Attach the custom routes after the OpenEnv application is created.
app.include_router(router)


def main() -> None:
    """Run the FastAPI app directly with uvicorn.

    This function parses command-line arguments and matches the names used
    by uvicorn to keep the developer experience consistent.
    """

    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the OpenEnv server.")
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "localhost"),
        help="Interface to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="TCP port to bind to"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", "1")),
        help="Number of worker processes"
    )

    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
