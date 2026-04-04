"""FastAPI application for the Python PR review OpenEnv."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from openenv.core.env_server.http_server import create_app

try:
    from models import (
        HealthResponse,
        PythonReviewAction,
        PythonReviewObservation,
        PythonReviewState,
        TaskDescriptor,
        TaskGrade,
        TaskSubmission,
        TaskSummary,
    )
    from server.code_review_environment import PythonEnvironment
except ModuleNotFoundError:  # pragma: no cover
    from ..models import (
        HealthResponse,
        PythonReviewAction,
        PythonReviewObservation,
        PythonReviewState,
        TaskDescriptor,
        TaskGrade,
        TaskSubmission,
        TaskSummary,
    )
    from .code_review_environment import PythonEnvironment


MAX_CONCURRENT_ENVS = int(os.getenv("MAX_CONCURRENT_ENVS", "16"))

python_env = PythonEnvironment()
app = create_app(
    PythonEnvironment,
    PythonReviewAction,
    PythonReviewObservation,
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)
router = APIRouter(tags=["python-pr-review"])


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect the root page to generated API docs."""

    return RedirectResponse(url="/docs")


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Lightweight deployment health route."""

    return HealthResponse(task_count=len(python_env.list_task_summaries()))


@router.get("/tasks", response_model=list[TaskSummary])
def list_tasks() -> list[TaskSummary]:
    """Return the three benchmark tasks in public form."""

    return python_env.list_task_summaries()


@router.get("/tasks/{task_id}", response_model=TaskDescriptor)
def get_task(task_id: str) -> TaskDescriptor:
    """Return one public task descriptor."""

    try:
        return python_env.get_task(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/tasks/{task_id}/grade", response_model=TaskGrade)
def grade_task(task_id: str, payload: TaskSubmission) -> TaskGrade:
    """Expose the deterministic grader for offline checks."""

    try:
        return python_env.grade_task_submission(task_id=task_id, findings=payload.findings)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/state", response_model=PythonReviewState)
def post_state() -> PythonReviewState:
    """Mirror the GET /state endpoint for clients that prefer POST."""

    return python_env.state


app.include_router(router)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI app with uvicorn."""

    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", host), port=int(os.getenv("PORT", str(port))))


if __name__ == "__main__":
    main()

