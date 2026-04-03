from models import PythonReviewAction, ReviewFinding
from server.code_review_environment import PythonEnvironment


def test_reset_cycles_tasks_in_order():
    env = PythonEnvironment()

    first = env.reset()
    second = env.reset()
    third = env.reset()

    assert first.task.task_id == "py-review-easy"
    assert second.task.task_id == "py-review-medium"
    assert third.task.task_id == "py-review-hard"


def test_partial_progress_reward_is_positive_for_new_match():
    env = PythonEnvironment()
    env.reset()

    action = PythonReviewAction(
        operation="submit_findings",
        findings=[
            ReviewFinding(
                title="Avoid eval on untrusted input",
                line=2,
                category="security",
                severity="critical",
                rationale="eval can execute attacker-controlled code.",
                recommendation="Use json.loads instead.",
            )
        ],
    )
    observation = env.step(action)

    assert observation.reward > 0
    assert observation.evaluation.matched_findings >= 1
    assert observation.done is False


def test_finalize_perfect_easy_task_passes():
    env = PythonEnvironment()
    env.reset()

    action = PythonReviewAction(
        operation="finalize",
        findings=[
            ReviewFinding(
                title="Avoid eval on untrusted configuration data",
                line=2,
                category="security",
                severity="critical",
                rationale="eval executes arbitrary code and is unsafe here.",
                recommendation="Use json.loads or ast.literal_eval.",
            ),
            ReviewFinding(
                title="Default count of zero causes a division by zero",
                line=5,
                category="bug",
                severity="warning",
                rationale="count defaults to zero and the division will crash.",
                recommendation="Validate count before dividing.",
            ),
        ],
    )
    observation = env.step(action)

    assert observation.done is True
    assert observation.score >= 0.8
    assert observation.evaluation.passed is True
