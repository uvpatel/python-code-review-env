"""FastAPI app for the Python code-review environment."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

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

# OpenEnv's server in your installed version expects an environment class/factory,
# not a pre-instantiated environment object.
python_env = PythonEnvironment()
app = create_app(PythonEnvironment, PythonReviewAction, PythonReviewObservation)
router = APIRouter(tags=["python-env"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    tasks = python_env.list_tasks()
    active_episode_id = None
    active_task_id = tasks[0].task_id if tasks else None
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
    return python_env.list_tasks()


@router.get("/tasks/{task_id}", response_model=TaskDescriptor)
def get_task(task_id: str) -> TaskDescriptor:
    try:
        return python_env.get_task(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/tasks/{task_id}/grade", response_model=TaskEvaluation)
def grade_task(task_id: str, payload: PythonReviewAction) -> TaskEvaluation:
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
    return python_env.direct_review(code=request.code, context=request.context)


@router.get("/history", response_model=list[EpisodeRecord])
def get_history() -> list[EpisodeRecord]:
    return python_env.get_history()


@router.delete("/history", response_model=DeleteResponse)
def clear_history() -> DeleteResponse:
    python_env.clear_history()
    return DeleteResponse(detail="history cleared")


@router.get("/config", response_model=PythonEnvConfig)
def get_config() -> PythonEnvConfig:
    return python_env.config


@router.put("/config", response_model=PythonEnvConfig)
def update_config(config: PythonEnvConfig) -> PythonEnvConfig:
    python_env.update_config(config)
    return python_env.config


app.include_router(router)


def main(host: str = "localhost", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
