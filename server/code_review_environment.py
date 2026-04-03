"""Deterministic Python code-review environment."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import (
        EpisodeRecord,
        PythonEnvConfig,
        PythonReviewAction,
        PythonReviewObservation,
        TaskEvaluation,
    )
    from server.grading import evaluate_submission
    from server.static_review import build_direct_review_response
    from server.task_bank import TaskSpec, get_task_by_id, get_task_descriptors
except ModuleNotFoundError:  # pragma: no cover
    from ..models import (
        EpisodeRecord,
        PythonEnvConfig,
        PythonReviewAction,
        PythonReviewObservation,
        TaskEvaluation,
    )
    from .grading import evaluate_submission
    from .static_review import build_direct_review_response
    from .task_bank import TaskSpec, get_task_by_id, get_task_descriptors


class PythonEnvironment(Environment):
    """Environment for benchmark-style Python code review."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config: Optional[PythonEnvConfig] = None):
        self.config = config or PythonEnvConfig()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_descriptors = get_task_descriptors(self.config.max_steps_per_task)
        self._task_cursor = -1
        self._current_task: Optional[TaskSpec] = None
        self._submitted_findings = []
        self._seen_fingerprints: Set[str] = set()
        self._matched_ids: Set[str] = set()
        self._false_positives = 0
        self._duplicate_findings = 0
        self._hints_used = 0
        self._best_patch_score = 0.0
        self._done = False
        self._history: List[EpisodeRecord] = []

    def reset(self) -> PythonReviewObservation:
        self._task_cursor = (self._task_cursor + 1) % len(self.config.task_order)
        task_id = self.config.task_order[self._task_cursor]
        self._current_task = get_task_by_id(task_id)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._submitted_findings = []
        self._seen_fingerprints = set()
        self._matched_ids = set()
        self._false_positives = 0
        self._duplicate_findings = 0
        self._hints_used = 0
        self._best_patch_score = 0.0
        self._done = False
        self._upsert_history_record(active=True)
        return self._build_observation(
            feedback=(
                "Review the Python snippet. Submit structured findings, request a hint, "
                "or finalize when you are satisfied."
            ),
            reward=0.0,
            review_time_ms=0.0,
        )

    def step(self, action: PythonReviewAction) -> PythonReviewObservation:
        if self._current_task is None:
            return self.reset()

        self._state.step_count += 1
        started = time.perf_counter()

        if self._done:
            return self._build_observation(
                feedback="Episode already completed. Call reset() to move to the next task.",
                reward=-0.1,
                review_time_ms=(time.perf_counter() - started) * 1000.0,
            )

        previous_score = self._current_evaluation().score
        feedback_lines: List[str] = []
        reward = 0.0

        if action.operation == "request_hint":
            feedback_lines.append(self._next_hint())
            reward -= self.config.hint_penalty
        else:
            grade = evaluate_submission(
                task=self._current_task,
                findings=action.findings,
                patched_code=action.patched_code,
                prior_matched_ids=self._matched_ids,
                prior_fingerprints=self._seen_fingerprints,
            )
            self._matched_ids.update(grade.newly_matched_ids)
            self._seen_fingerprints.update(grade.accepted_fingerprints)
            self._submitted_findings.extend(action.findings)
            self._false_positives += grade.false_positives
            self._duplicate_findings += grade.duplicate_findings

            old_patch_score = self._best_patch_score
            self._best_patch_score = max(self._best_patch_score, grade.patch_score)
            patch_gain = self._best_patch_score - old_patch_score

            if grade.newly_matched_ids:
                feedback_lines.append(
                    f"Matched {len(grade.newly_matched_ids)} new rubric item(s): "
                    + ", ".join(grade.newly_matched_ids)
                )
            if grade.false_positives:
                feedback_lines.append(
                    f"{grade.false_positives} submission(s) did not match the rubric."
                )
            if grade.duplicate_findings:
                feedback_lines.append(
                    f"{grade.duplicate_findings} duplicate finding(s) were ignored."
                )
            if action.patched_code:
                feedback_lines.append(
                    f"Patch quality score: {self._best_patch_score:.2f}."
                )
            if action.note:
                feedback_lines.append(f"Reviewer note recorded: {action.note}")
            if not feedback_lines:
                feedback_lines.append("No new rubric matches yet. Tighten your findings.")

            current_score = self._current_evaluation().score
            reward += current_score - previous_score
            reward += patch_gain * self.config.patch_bonus_multiplier
            reward -= grade.false_positives * self.config.false_positive_penalty
            reward -= grade.duplicate_findings * self.config.duplicate_penalty

        if action.operation == "finalize":
            self._done = True
            if self._current_evaluation().passed:
                reward += 0.1
                feedback_lines.append("Final submission passed the task threshold.")
            else:
                reward -= 0.05
                feedback_lines.append("Final submission did not reach the passing threshold.")

        if self._state.step_count >= self.config.max_steps_per_task and not self._done:
            self._done = True
            feedback_lines.append("Maximum steps reached. Episode closed automatically.")

        self._upsert_history_record(active=not self._done)
        return self._build_observation(
            feedback=" ".join(feedback_lines).strip(),
            reward=reward,
            review_time_ms=(time.perf_counter() - started) * 1000.0,
        )

    def direct_review(self, code: str, context: Optional[str] = None):
        return build_direct_review_response(code=code, context=context)

    def list_tasks(self):
        return list(self._task_descriptors)

    def get_task(self, task_id: str):
        return get_task_by_id(task_id).to_descriptor(self.config.max_steps_per_task)

    def grade_task_submission(self, task_id: str, findings, patched_code: Optional[str] = None):
        task = get_task_by_id(task_id)
        return evaluate_submission(
            task=task, findings=findings, patched_code=patched_code
        ).evaluation

    def get_history(self) -> List[EpisodeRecord]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def update_config(self, config: PythonEnvConfig) -> None:
        self.config = config
        self._task_descriptors = get_task_descriptors(self.config.max_steps_per_task)
        if not self.config.task_order:
            self.config.task_order = [task.task_id for task in self._task_descriptors]

    def _current_evaluation(self) -> TaskEvaluation:
        if self._current_task is None:
            return TaskEvaluation()
        return evaluate_submission(
            task=self._current_task,
            findings=self._submitted_findings,
            patched_code=None,
            prior_matched_ids=self._matched_ids,
            prior_fingerprints=self._seen_fingerprints,
            force_patch_score=self._best_patch_score,
            use_existing_matches=True,
            false_positives=self._false_positives,
            duplicate_findings=self._duplicate_findings,
        ).evaluation

    def _build_observation(
        self, feedback: str, reward: float, review_time_ms: float
    ) -> PythonReviewObservation:
        if self._current_task is None:
            task_descriptor = self._task_descriptors[0]
            evaluation = TaskEvaluation()
        else:
            task_descriptor = self._current_task.to_descriptor(self.config.max_steps_per_task)
            evaluation = self._current_evaluation()

        return PythonReviewObservation(
            task=task_descriptor,
            instructions=(
                "Identify real issues in the code. Submit structured findings with line "
                "numbers when possible. Request a hint only when needed."
            ),
            feedback=feedback,
            submitted_findings=list(self._submitted_findings),
            hints_used=self._hints_used,
            attempts_remaining=max(task_descriptor.max_steps - self._state.step_count, 0),
            evaluation=evaluation,
            score=evaluation.score,
            review_time_ms=review_time_ms,
            done=self._done,
            reward=max(min(reward, 1.0), -1.0),
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "matched_findings": evaluation.matched_findings,
                "total_findings": evaluation.total_findings,
                "success_threshold": task_descriptor.success_threshold,
            },
        )

    def _next_hint(self) -> str:
        if self._current_task is None:
            return "No active task."
        if self._hints_used >= len(self._current_task.hints):
            return "No hints remaining for this task."
        hint = self._current_task.hints[self._hints_used]
        self._hints_used += 1
        return f"Hint {self._hints_used}: {hint}"

    def _upsert_history_record(self, active: bool) -> None:
        if self._current_task is None:
            return
        evaluation = self._current_evaluation()
        now = datetime.now(timezone.utc).isoformat()
        record = EpisodeRecord(
            episode_id=self._state.episode_id,
            task_id=self._current_task.task_id,
            difficulty=self._current_task.difficulty,
            title=self._current_task.title,
            final_score=evaluation.score,
            passed=evaluation.passed,
            steps_taken=self._state.step_count,
            hints_used=self._hints_used,
            matched_findings=evaluation.matched_findings,
            total_findings=evaluation.total_findings,
            false_positives=self._false_positives,
            duplicate_findings=self._duplicate_findings,
            status="active" if active else "completed",
            created_at=now,
            updated_at=now,
        )
        for index, existing in enumerate(self._history):
            if existing.episode_id == record.episode_id:
                record.created_at = existing.created_at
                self._history[index] = record
                break
        else:
            self._history.append(record)

        if len(self._history) > self.config.max_history_entries:
            self._history = self._history[-self.config.max_history_entries :]

    @property
    def state(self) -> State:
        return self._state


CodeReviewEnvironment = PythonEnvironment
