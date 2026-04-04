from models import ReviewFinding
from server.code_review_environment import PythonEnvironment
from server.grading import grade_findings
from server.task_bank import get_task


def test_canonical_perfect_submission_returns_one_for_easy():
    task = get_task("py-pr-review-easy")
    score = grade_findings(
        task,
        [
            ReviewFinding(
                file_path="src/notifications/retry.py",
                line=7,
                category="bug",
                severity="warning",
                title="Zero base_delay still divides",
                explanation="A zero value slips through the guard and causes division by zero.",
                suggested_fix="Treat zero like other invalid base_delay values.",
            )
        ],
    )

    assert score.score == 1.0


def test_empty_submission_returns_zero_for_hard():
    task = get_task("py-pr-review-hard")
    score = grade_findings(task, [])

    assert score.score == 0.0


def test_mixed_submission_returns_exact_partial_score_for_medium():
    task = get_task("py-pr-review-medium")
    score = grade_findings(
        task,
        [
            ReviewFinding(
                file_path="app/billing/invoice_service.py",
                line=8,
                category="bug",
                severity="warning",
                title="Discount total is computed but never charged",
                explanation="The code calculates a discounted total but still charges the original amount.",
                suggested_fix="Pass total to gateway.charge instead of amount_cents.",
            ),
            ReviewFinding(
                file_path="app/billing/invoice_service.py",
                line=4,
                category="performance",
                severity="warning",
                title="Use caching here",
                explanation="This line should probably cache something.",
                suggested_fix="Add memoization.",
            ),
        ],
    )

    assert score.score == 0.5


def test_offline_grader_route_matches_environment_logic():
    env = PythonEnvironment()
    grade = env.grade_task_submission(
        "py-pr-review-medium",
        [
            ReviewFinding(
                file_path="tests/test_invoice_service.py",
                line=11,
                category="testing",
                severity="warning",
                title="Missing coupon test coverage",
                explanation="There is no test for a discounted order with coupon_code.",
                suggested_fix="Add a coupon test that asserts the discounted charge amount.",
            )
        ],
    )

    assert grade.score == 0.4
