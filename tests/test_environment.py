from models import PythonReviewAction, ReviewFinding
from server.code_review_environment import PythonEnvironment


def test_reset_cycles_tasks_in_order():
    env = PythonEnvironment()

    first = env.reset()
    second = env.reset()
    third = env.reset()

    assert first.task_id == "py-pr-review-easy"
    assert second.task_id == "py-pr-review-medium"
    assert third.task_id == "py-pr-review-hard"


def test_invalid_action_penalizes_without_corrupting_state():
    env = PythonEnvironment()
    env.reset()

    observation = env.step(PythonReviewAction(operation="read_file"))

    assert observation.reward < 0
    assert observation.last_action_status.startswith("Invalid action")
    assert observation.metadata["submitted_finding_count"] == 0
    assert observation.done is False


def test_duplicate_finding_is_penalized_once_per_repeat():
    env = PythonEnvironment()
    env.reset()
    finding = ReviewFinding(
        file_path="src/notifications/retry.py",
        line=7,
        category="bug",
        severity="warning",
        title="Zero base_delay still divides",
        explanation="A zero base_delay reaches the division and crashes.",
        suggested_fix="Reject or handle zero before dividing.",
    )

    first = env.step(PythonReviewAction(operation="add_finding", finding=finding))
    second = env.step(PythonReviewAction(operation="add_finding", finding=finding))

    assert first.reward > 0
    assert second.reward < 0
    assert second.metadata["duplicate_findings"] == 1


def test_submit_review_ends_episode_with_bounded_score():
    env = PythonEnvironment()
    env.reset()
    env.step(
        PythonReviewAction(
            operation="add_finding",
            finding=ReviewFinding(
                file_path="src/notifications/retry.py",
                line=7,
                category="bug",
                severity="warning",
                title="Zero base_delay still divides",
                explanation="When base_delay is zero the division still executes.",
                suggested_fix="Guard base_delay <= 0 before dividing.",
            ),
        )
    )

    observation = env.step(PythonReviewAction(operation="submit_review"))

    assert observation.done is True
    assert 0.0 <= observation.score <= 1.0
    assert observation.last_action_status.startswith("Review submitted")
