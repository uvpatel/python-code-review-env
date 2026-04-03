"""Client for the Python code-review environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import (
        PythonReviewAction,
        PythonReviewObservation,
        ReviewFinding,
        TaskDescriptor,
        TaskEvaluation,
    )
except ImportError:  # pragma: no cover
    from models import (  # type: ignore
        PythonReviewAction,
        PythonReviewObservation,
        ReviewFinding,
        TaskDescriptor,
        TaskEvaluation,
    )


class PythonEnv(EnvClient[PythonReviewAction, PythonReviewObservation, State]):
    """OpenEnv client for the Python code-review benchmark."""

    def _step_payload(self, action: PythonReviewAction) -> Dict:
        return {
            "operation": action.operation,
            "findings": [finding.model_dump() for finding in action.findings],
            "patched_code": action.patched_code,
            "note": action.note,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PythonReviewObservation]:
        obs = payload.get("observation", {})
        observation = PythonReviewObservation(
            task=TaskDescriptor(**obs["task"]),
            instructions=obs.get("instructions", ""),
            feedback=obs.get("feedback", ""),
            submitted_findings=[
                ReviewFinding(**finding) for finding in obs.get("submitted_findings", [])
            ],
            hints_used=obs.get("hints_used", 0),
            attempts_remaining=obs.get("attempts_remaining", 0),
            evaluation=TaskEvaluation(**obs.get("evaluation", {})),
            score=obs.get("score", 0.0),
            review_time_ms=obs.get("review_time_ms", 0.0),
            done=payload.get("done", obs.get("done", False)),
            reward=payload.get("reward", obs.get("reward")),
            metadata=obs.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs.get("reward")),
            done=payload.get("done", obs.get("done", False)),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


CodeReviewEnv = PythonEnv
MyEnv = PythonEnv
