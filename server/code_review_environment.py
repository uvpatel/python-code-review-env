"""OpenEnv environment for deterministic Python pull-request review."""

from __future__ import annotations

from typing import List, Optional, Set
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from models import (
        PythonReviewAction,
        PythonReviewObservation,
        PythonReviewReward,
        PythonReviewState,
        ReviewFinding,
        ReviewHistoryEntry,
        TaskDescriptor,
        TaskGrade,
        reward_metadata,
    )
    from server.grading import (
        FALSE_POSITIVE_PENALTY,
        DUPLICATE_PENALTY,
        finding_fingerprint,
        match_finding,
        score_task,
    )
    from server.task_bank import TaskSpec, get_task, list_task_descriptors, list_task_summaries, task_ids
except ModuleNotFoundError:  # pragma: no cover
    from ..models import (
        PythonReviewAction,
        PythonReviewObservation,
        PythonReviewReward,
        PythonReviewState,
        ReviewFinding,
        ReviewHistoryEntry,
        TaskDescriptor,
        TaskGrade,
        reward_metadata,
    )
    from .grading import (
        FALSE_POSITIVE_PENALTY,
        DUPLICATE_PENALTY,
        finding_fingerprint,
        match_finding,
        score_task,
    )
    from .task_bank import TaskSpec, get_task, list_task_descriptors, list_task_summaries, task_ids


READ_FILE_REWARD = 0.05
REPEATED_FILE_PENALTY = 0.03
INVALID_ACTION_PENALTY = 0.10
STEP_EFFICIENCY_PENALTY = 0.01


class PythonEnvironment(Environment[PythonReviewAction, PythonReviewObservation, PythonReviewState]):
    """Simulate a realistic PR-review workflow for Python changes."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._task_order = list(task_ids())
        self._task_cursor = -1
        self._current_task: Optional[TaskSpec] = None
        self._state = PythonReviewState()
        self._opened_files: List[str] = []
        self._visible_sections: List[str] = []
        self._submitted_findings: List[ReviewFinding] = []
        self._review_history: List[ReviewHistoryEntry] = []
        self._seen_fingerprints: Set[str] = set()
        self._matched_issue_ids: Set[str] = set()
        self._false_positives = 0
        self._duplicate_findings = 0
        self._last_reward = PythonReviewReward(value=0.0, reason="Environment initialized.")
        self._last_status = "Call reset() to start reviewing a task."
        self._done = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **_: object,
    ) -> PythonReviewObservation:
        """Reset the environment to a clean PR-review episode."""

        del seed  # The benchmark is deterministic; seed is accepted for interface compatibility.
        if task_id:
            self._current_task = get_task(task_id)
            self._task_cursor = self._task_order.index(task_id)
        else:
            self._task_cursor = (self._task_cursor + 1) % len(self._task_order)
            self._current_task = get_task(self._task_order[self._task_cursor])

        self._opened_files = []
        self._visible_sections = [self._current_task.visible_diff]
        self._submitted_findings = []
        self._review_history = []
        self._seen_fingerprints = set()
        self._matched_issue_ids = set()
        self._false_positives = 0
        self._duplicate_findings = 0
        self._done = False
        self._last_reward = PythonReviewReward(value=0.0, reason="Episode reset.")
        self._last_status = (
            "Review the diff, open files for context, add findings, then submit the review."
        )
        self._state = PythonReviewState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._current_task.task_id,
            difficulty=self._current_task.difficulty,
            attempts_remaining=self._current_task.max_steps,
            opened_files=[],
            submitted_findings=[],
            review_history=[],
            score=0.0,
            done=False,
        )
        return self._build_observation()

    def step(
        self,
        action: PythonReviewAction,
        timeout_s: Optional[float] = None,
        **_: object,
    ) -> PythonReviewObservation:
        """Apply one review action and return the updated observation."""

        del timeout_s
        if self._current_task is None:
            return self.reset()
        if self._done:
            self._last_reward = PythonReviewReward(
                value=-INVALID_ACTION_PENALTY,
                invalid_action_penalty=INVALID_ACTION_PENALTY,
                reason="Episode already finished.",
            )
            self._last_status = "Episode already finished. Call reset() for the next task."
            return self._build_observation()

        self._state.step_count += 1
        reward = PythonReviewReward(
            value=-STEP_EFFICIENCY_PENALTY,
            efficiency_penalty=STEP_EFFICIENCY_PENALTY,
            reason="Processed action.",
        )
        status = ""

        if action.operation == "read_file":
            reward, status = self._handle_read_file(action)
        elif action.operation == "add_finding":
            reward, status = self._handle_add_finding(action)
        elif action.operation in {"submit_review", "finish"}:
            reward, status = self._handle_finish(action.operation)
        else:  # pragma: no cover
            reward = PythonReviewReward(
                value=-INVALID_ACTION_PENALTY,
                invalid_action_penalty=INVALID_ACTION_PENALTY,
                reason=f"Unsupported operation: {action.operation}",
            )
            status = f"Unsupported operation: {action.operation}"

        self._last_reward = reward
        self._last_status = status
        self._state.attempts_remaining = max(
            self._current_task.max_steps - self._state.step_count, 0
        )

        if self._state.step_count >= self._current_task.max_steps and not self._done:
            final_grade = self._current_grade()
            terminal = PythonReviewReward(
                value=max(0.0, final_grade.score - STEP_EFFICIENCY_PENALTY),
                efficiency_penalty=STEP_EFFICIENCY_PENALTY,
                reason="Step budget exhausted; episode closed automatically.",
            )
            self._done = True
            self._last_reward = terminal
            self._last_status = "Maximum steps reached. Review submitted automatically."

        self._state.done = self._done
        self._state.score = self._current_grade().score
        return self._build_observation()

    @property
    def state(self) -> PythonReviewState:
        """Return the current non-secret environment state."""

        return self._state.model_copy(deep=True)

    def list_tasks(self) -> List[TaskDescriptor]:
        """Return all public task descriptors."""

        return list_task_descriptors()

    def list_task_summaries(self):
        """Return lightweight task summaries."""

        return list_task_summaries()

    def get_task(self, task_id: str) -> TaskDescriptor:
        """Return one public task descriptor."""

        return get_task(task_id).to_descriptor()

    def grade_task_submission(self, task_id: str, findings: List[ReviewFinding]) -> TaskGrade:
        """Offline deterministic grader for tests and debugging."""

        task = get_task(task_id)
        matched_issue_ids: Set[str] = set()
        seen_fingerprints: Set[str] = set()
        false_positives = 0
        duplicate_findings = 0
        for finding in findings:
            result = match_finding(
                finding=finding,
                task=task,
                matched_issue_ids=matched_issue_ids,
                seen_fingerprints=seen_fingerprints,
            )
            fingerprint = finding_fingerprint(finding)
            if result.duplicate:
                duplicate_findings += 1
                continue
            seen_fingerprints.add(fingerprint)
            if result.issue_id is None:
                false_positives += 1
                continue
            matched_issue_ids.add(result.issue_id)
        return score_task(
            task=task,
            matched_issue_ids=matched_issue_ids,
            false_positives=false_positives,
            duplicate_findings=duplicate_findings,
        )

    def _handle_read_file(self, action: PythonReviewAction) -> tuple[PythonReviewReward, str]:
        if not action.path:
            return (
                PythonReviewReward(
                    value=-INVALID_ACTION_PENALTY,
                    invalid_action_penalty=INVALID_ACTION_PENALTY,
                    reason="read_file requires a path.",
                ),
                "Invalid action: read_file requires a path.",
            )
        if action.path not in self._current_task.file_contents:
            return (
                PythonReviewReward(
                    value=-INVALID_ACTION_PENALTY,
                    invalid_action_penalty=INVALID_ACTION_PENALTY,
                    reason=f"{action.path} is not available in this task.",
                ),
                f"Invalid action: {action.path} is not available.",
            )
        if action.path in self._opened_files:
            self._append_history(action.operation, f"Re-opened {action.path}.")
            return (
                PythonReviewReward(
                    value=-REPEATED_FILE_PENALTY,
                    efficiency_penalty=REPEATED_FILE_PENALTY,
                    reason=f"{action.path} was already opened.",
                ),
                f"{action.path} was already opened; no new context added.",
            )

        self._opened_files.append(action.path)
        self._visible_sections.append(
            "\n".join(
                [
                    "",
                    f"# Full file: {action.path}",
                    self._current_task.file_contents[action.path],
                ]
            )
        )
        self._append_history(action.operation, f"Opened {action.path} for full context.")
        self._sync_state()
        return (
            PythonReviewReward(
                value=READ_FILE_REWARD - STEP_EFFICIENCY_PENALTY,
                matched_progress=READ_FILE_REWARD,
                efficiency_penalty=STEP_EFFICIENCY_PENALTY,
                reason=f"Opened {action.path}.",
            ),
            f"Opened {action.path}. Full file content is now visible.",
        )

    def _handle_add_finding(self, action: PythonReviewAction) -> tuple[PythonReviewReward, str]:
        if action.finding is None:
            return (
                PythonReviewReward(
                    value=-INVALID_ACTION_PENALTY,
                    invalid_action_penalty=INVALID_ACTION_PENALTY,
                    reason="add_finding requires a finding payload.",
                ),
                "Invalid action: add_finding requires a finding payload.",
            )

        result = match_finding(
            finding=action.finding,
            task=self._current_task,
            matched_issue_ids=self._matched_issue_ids,
            seen_fingerprints=self._seen_fingerprints,
        )
        fingerprint = finding_fingerprint(action.finding)
        self._submitted_findings.append(action.finding)

        if result.duplicate:
            self._duplicate_findings += 1
            self._append_history("add_finding", f"Duplicate finding ignored for {action.finding.file_path}.")
            self._sync_state()
            return (
                PythonReviewReward(
                    value=-(DUPLICATE_PENALTY + STEP_EFFICIENCY_PENALTY),
                    efficiency_penalty=STEP_EFFICIENCY_PENALTY,
                    reason="Duplicate finding.",
                ),
                "Duplicate finding: this issue was already submitted.",
            )

        self._seen_fingerprints.add(fingerprint)
        if result.issue_id is None:
            self._false_positives += 1
            self._append_history(
                "add_finding",
                f"Unsupported finding on {action.finding.file_path}:{action.finding.line or '?'}",
            )
            self._sync_state()
            return (
                PythonReviewReward(
                    value=-(FALSE_POSITIVE_PENALTY + STEP_EFFICIENCY_PENALTY),
                    false_positive_penalty=FALSE_POSITIVE_PENALTY,
                    efficiency_penalty=STEP_EFFICIENCY_PENALTY,
                    reason="Finding did not match the hidden rubric.",
                ),
                "Finding recorded, but it did not match a rubric issue.",
            )

        self._matched_issue_ids.add(result.issue_id)
        progress = self._current_issue_weight(result.issue_id)
        self._append_history(
            "add_finding",
            f"Matched rubric issue {result.issue_id} on {action.finding.file_path}.",
        )
        self._sync_state()
        return (
            PythonReviewReward(
                value=progress - STEP_EFFICIENCY_PENALTY,
                matched_progress=progress,
                efficiency_penalty=STEP_EFFICIENCY_PENALTY,
                reason=f"Matched rubric issue {result.issue_id}.",
            ),
            f"Accepted finding. It matched issue {result.issue_id}.",
        )

    def _handle_finish(self, operation: str) -> tuple[PythonReviewReward, str]:
        final_grade = self._current_grade()
        self._done = True
        self._append_history(operation, f"Submitted review with score {final_grade.score:.2f}.")
        self._sync_state()
        return (
            PythonReviewReward(
                value=max(0.0, final_grade.score - STEP_EFFICIENCY_PENALTY),
                efficiency_penalty=STEP_EFFICIENCY_PENALTY,
                reason=f"Review submitted with final score {final_grade.score:.2f}.",
            ),
            (
                f"Review submitted. Final score: {final_grade.score:.2f} "
                f"with {len(final_grade.matched_issue_ids)} matched issue(s)."
            ),
        )

    def _current_issue_weight(self, issue_id: str) -> float:
        for issue in self._current_task.rubric_issues:
            if issue.issue_id == issue_id:
                return issue.weight
        return 0.0

    def _current_grade(self) -> TaskGrade:
        return score_task(
            task=self._current_task,
            matched_issue_ids=self._matched_issue_ids,
            false_positives=self._false_positives,
            duplicate_findings=self._duplicate_findings,
        )

    def _append_history(self, operation: str, summary: str) -> None:
        self._review_history.append(
            ReviewHistoryEntry(
                step=self._state.step_count,
                operation=operation,
                summary=summary,
            )
        )

    def _sync_state(self) -> None:
        self._state.opened_files = list(self._opened_files)
        self._state.submitted_findings = list(self._submitted_findings)
        self._state.review_history = list(self._review_history)
        self._state.score = self._current_grade().score
        self._state.done = self._done

    def _build_observation(self) -> PythonReviewObservation:
        if self._current_task is None:  # pragma: no cover
            self.reset()

        self._sync_state()
        return PythonReviewObservation(
            task_id=self._current_task.task_id,
            difficulty=self._current_task.difficulty,
            goal=self._current_task.goal,
            repo_summary=self._current_task.repo_summary,
            changed_files=list(self._current_task.changed_files),
            visible_diff="\n".join(self._visible_sections),
            available_files=self._current_task.available_files,
            review_history=list(self._review_history),
            attempts_remaining=self._state.attempts_remaining,
            last_action_status=self._last_status,
            score=self._current_grade().score,
            reward_details=self._last_reward,
            done=self._done,
            reward=self._last_reward.value,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": self._state.step_count,
                "opened_files": list(self._opened_files),
                "submitted_finding_count": len(self._submitted_findings),
                "false_positives": self._false_positives,
                "duplicate_findings": self._duplicate_findings,
                "info": {
                    "reward": reward_metadata(self._last_reward),
                    "matched_issue_count": len(self._matched_issue_ids),
                    "possible_issue_count": len(self._current_task.rubric_issues),
                },
            },
        )


CodeReviewEnvironment = PythonEnvironment
