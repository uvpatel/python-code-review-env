# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Code review environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CodeReviewAction, CodeReviewObservation, ReviewIssue


class CodeReviewEnv(
    EnvClient[CodeReviewAction, CodeReviewObservation, State]
):
    """
    Client for the code review environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with CodeReviewEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     result = client.step(CodeReviewAction(code="x = 1"))
        ...     print(f"Issues: {len(result.observation.issues)}")
        ...     print(f"Score: {result.observation.score}")

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = CodeReviewEnv.from_docker_image("python_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(CodeReviewAction(code="def foo(): pass"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: CodeReviewAction) -> Dict:
        """
        Convert CodeReviewAction to JSON payload for step message.

        Args:
            action: CodeReviewAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "code": action.code,
            "language": action.language,
            "focus_areas": action.focus_areas,
            "context": action.context,
        }

    def _parse_result(self, payload: Dict) -> StepResult[CodeReviewObservation]:
        """
        Parse server response into StepResult[CodeReviewObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with CodeReviewObservation
        """
        obs_data = payload.get("observation", {})
        observation = CodeReviewObservation(
            original_code=obs_data.get("original_code", ""),
            language=obs_data.get("language", "python"),
            issues=[ReviewIssue(**issue) for issue in obs_data.get("issues", [])],
            summary=obs_data.get("summary", ""),
            score=obs_data.get("score", 0.0),
            improved_code=obs_data.get("improved_code"),
            review_time_ms=obs_data.get("review_time_ms", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


# Backward-compatible aliases for existing integrations
PythonEnv = CodeReviewEnv
MyEnv = CodeReviewEnv
