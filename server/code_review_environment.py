# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Code review environment implementation.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    CodeReviewAction,
    CodeReviewConfig,
    CodeReviewObservation,
    ReviewIssue,
)


class _LLMClientStub:
    """Lightweight placeholder that parses inline markers instead of calling a real LLM."""

    def review_code(
        self, code: str, language: str, focus_areas: List[str], context: Optional[str]
    ) -> Dict:
        issues = self._parse_markers(code)
        improved_code = self._suggest_improvement(code, issues)
        summary = self._create_summary(issues, focus_areas)
        return {"issues": issues, "summary": summary, "improved_code": improved_code}

    def _parse_markers(self, code: str) -> List[Dict]:
        parsed: List[Dict] = []
        for lineno, line in enumerate(code.splitlines(), start=1):
            if "REVIEW_ISSUE:" not in line:
                continue
            payload = line.split("REVIEW_ISSUE:", 1)[1]
            fields = {}
            for kv in payload.split(";"):
                if "=" not in kv:
                    continue
                key, value = kv.split("=", 1)
                fields[key.strip()] = value.strip().strip('"').strip("'")
            severity = fields.get("severity", "warning")
            category = fields.get("category", "bug")
            parsed.append(
                {
                    "severity": severity if severity in {"critical", "warning", "info"} else "warning",
                    "category": category
                    if category in {"bug", "security", "style", "performance"}
                    else "bug",
                    "line": int(fields.get("line", lineno)),
                    "message": fields.get("message", "Issue reported."),
                    "suggestion": fields.get("suggestion"),
                }
            )
        return parsed

    def _suggest_improvement(self, code: str, issues: List[Dict]) -> Optional[str]:
        if not issues:
            return None
        return code + "\n# REVIEW_SUGGESTION: " + " | ".join(issue["message"] for issue in issues)

    def _create_summary(self, issues: List[Dict], focus_areas: List[str]) -> str:
        if not issues:
            return "No issues detected."
        return f"Detected {len(issues)} issue(s) within focus areas {focus_areas}."


class CodeReviewEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config: Optional[CodeReviewConfig] = None):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.config = config or CodeReviewConfig()
        self.review_history: List[Dict] = []
        self.total_issues_found = 0
        self.session_score_avg = 0.0
        self._scores: List[float] = []
        self._llm = _LLMClientStub()

    def reset(self) -> CodeReviewObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self.review_history.clear()
        self.total_issues_found = 0
        self._scores.clear()
        self.session_score_avg = 0.0
        return CodeReviewObservation(
            original_code="",
            language="python",
            issues=[],
            summary="Code reviewer ready.",
            score=0.0,
            review_time_ms=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: CodeReviewAction) -> CodeReviewObservation:
        self._state.step_count += 1
        start = time.perf_counter()

        if not action.code.strip():
            reward = -1.0
            issues: List[ReviewIssue] = []
            summary = "No code provided."
            improved = None
        else:
            raw = self._llm.review_code(
                action.code, action.language, action.focus_areas, action.context
            )
            issues = [ReviewIssue(**issue) for issue in raw["issues"]]
            summary = raw["summary"]
            improved = raw.get("improved_code")
            reward = self._compute_reward(issues, bool(improved))
            self.total_issues_found += len(issues)

        duration_ms = (time.perf_counter() - start) * 1000.0
        score = self._score_from_issues(issues)
        self._scores.append(score)
        self.session_score_avg = sum(self._scores) / len(self._scores)

        observation = CodeReviewObservation(
            original_code=action.code,
            language=action.language,
            issues=issues,
            summary=summary,
            score=score,
            improved_code=improved,
            review_time_ms=duration_ms,
            done=False,
            reward=reward,
            metadata={
                "focus_areas": action.focus_areas,
                "session_score_avg": self.session_score_avg,
            },
        )
        self.review_history.append(
            {
                "job_id": str(uuid4()),
                "action": action,
                "observation": observation,
                "reward": reward,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        return observation

    def _compute_reward(self, issues: List[ReviewIssue], improved: bool) -> float:
        weights = {"critical": 2.0, "warning": 1.0, "info": 0.5}
        reward = sum(weights.get(issue.severity, 0.5) for issue in issues)
        if improved:
            reward += 1.0
        if not issues:
            reward -= 0.5
        return max(reward, -1.0)

    @staticmethod
    def _score_from_issues(issues: List[ReviewIssue]) -> float:
        if not issues:
            return 9.5
        base_score = 10 - len(issues)
        return max(0.0, min(10.0, base_score))

    def get_history(self) -> List[Dict]:
        return list(self.review_history)

    def clear_history(self) -> None:
        self.review_history.clear()

    def update_config(self, config: CodeReviewConfig) -> None:
        self.config = config

    @property
    def state(self) -> State:
        return self._state
