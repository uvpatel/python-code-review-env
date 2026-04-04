"""Client for the Python PR review environment."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        PythonReviewAction,
        PythonReviewObservation,
        PythonReviewReward,
        PythonReviewState,
        ReviewFinding,
        ReviewHistoryEntry,
    )
except ImportError:  # pragma: no cover
    from models import (  # type: ignore
        PythonReviewAction,
        PythonReviewObservation,
        PythonReviewReward,
        PythonReviewState,
        ReviewFinding,
        ReviewHistoryEntry,
    )


class PythonEnv(
    EnvClient[PythonReviewAction, PythonReviewObservation, PythonReviewState]
):
    """OpenEnv HTTP client for the Python PR review benchmark."""

    def _step_payload(self, action: PythonReviewAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[PythonReviewObservation]:
        obs = payload.get("observation", {})
        observation = PythonReviewObservation(
            task_id=obs["task_id"],
            difficulty=obs["difficulty"],
            goal=obs["goal"],
            repo_summary=obs["repo_summary"],
            changed_files=obs.get("changed_files", []),
            visible_diff=obs.get("visible_diff", ""),
            available_files=obs.get("available_files", []),
            review_history=[
                ReviewHistoryEntry(**entry) for entry in obs.get("review_history", [])
            ],
            attempts_remaining=obs.get("attempts_remaining", 0),
            last_action_status=obs.get("last_action_status", ""),
            score=obs.get("score", 0.0),
            reward_details=PythonReviewReward(**obs.get("reward_details", {"value": 0.0})),
            done=payload.get("done", obs.get("done", False)),
            reward=payload.get("reward", obs.get("reward")),
            metadata=obs.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs.get("reward")),
            done=payload.get("done", obs.get("done", False)),
        )

    def _parse_state(self, payload: Dict) -> PythonReviewState:
        return PythonReviewState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id"),
            difficulty=payload.get("difficulty"),
            attempts_remaining=payload.get("attempts_remaining", 0),
            opened_files=payload.get("opened_files", []),
            submitted_findings=[
                ReviewFinding(**finding) for finding in payload.get("submitted_findings", [])
            ],
            review_history=[
                ReviewHistoryEntry(**entry) for entry in payload.get("review_history", [])
            ],
            score=payload.get("score", 0.0),
            done=payload.get("done", False),
        )


CodeReviewEnv = PythonEnv
MyEnv = PythonEnv

