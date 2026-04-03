"""Tests for server/grading.py – evaluate_submission and helpers."""

import pytest

from models import ReviewFinding, TaskEvaluation
from server.grading import (
    _fingerprint,
    _grade_patch_score,
    _phrase_overlap,
    _reference_similarity,
    _tokens,
    evaluate_submission,
)
from server.task_bank import ReferenceFinding, TaskSpec, get_task_by_id


# ---------------------------------------------------------------------------
# Helper: build a minimal TaskSpec with one reference finding
# ---------------------------------------------------------------------------

def _make_task(
    rule_id: str = "avoid-eval",
    category: str = "security",
    severity: str = "critical",
    line: int = 2,
    success_threshold: float = 0.75,
) -> TaskSpec:
    ref = ReferenceFinding(
        finding_id=rule_id,
        title="Avoid eval",
        category=category,
        severity=severity,
        line=line,
        aliases=("eval execution", "code injection"),
        recommendation="Use json.loads.",
        weight=1.0,
        rule_id=rule_id,
    )
    return TaskSpec(
        task_id="test-task",
        difficulty="easy",
        title="Test",
        objective="Test",
        code="eval(x)",
        reference_findings=(ref,),
        success_threshold=success_threshold,
    )


def _good_finding(rule_id: str = "avoid-eval") -> ReviewFinding:
    return ReviewFinding(
        title="Avoid eval on untrusted input",
        line=2,
        category="security",
        severity="critical",
        rationale="eval executes arbitrary code.",
        recommendation="Use json.loads instead.",
        rule_id=rule_id,
    )


# ---------------------------------------------------------------------------
# _tokens
# ---------------------------------------------------------------------------

def test_tokens_strips_stopwords():
    tokens = _tokens("eval is a security risk")
    assert "is" not in tokens
    assert "a" not in tokens
    assert "eval" in tokens
    assert "security" in tokens


def test_tokens_lowercases():
    tokens = _tokens("EVAL Security")
    assert "eval" in tokens
    assert "security" in tokens


def test_tokens_empty_string():
    assert _tokens("") == set()


# ---------------------------------------------------------------------------
# _phrase_overlap
# ---------------------------------------------------------------------------

def test_phrase_overlap_exact_match_returns_one():
    tokens = {"eval", "security", "risk"}
    assert _phrase_overlap(tokens, "eval security risk") == 1.0


def test_phrase_overlap_partial_returns_fraction():
    tokens = {"eval", "security"}
    score = _phrase_overlap(tokens, "eval security risk")
    assert 0.0 < score < 1.0


def test_phrase_overlap_empty_phrase_returns_zero():
    tokens = {"eval"}
    assert _phrase_overlap(tokens, "") == 0.0


def test_phrase_overlap_no_overlap_returns_zero():
    tokens = {"unrelated"}
    assert _phrase_overlap(tokens, "eval security") == 0.0


# ---------------------------------------------------------------------------
# _reference_similarity
# ---------------------------------------------------------------------------

def test_reference_similarity_high_for_matching_finding():
    task = _make_task()
    ref = task.reference_findings[0]
    finding = _good_finding()

    score = _reference_similarity(finding, ref)
    assert score >= 0.55


def test_reference_similarity_low_for_wrong_category():
    task = _make_task()
    ref = task.reference_findings[0]
    finding = ReviewFinding(
        title="Something unrelated",
        line=99,
        category="style",
        severity="info",
        rationale="no relation",
    )
    score = _reference_similarity(finding, ref)
    assert score < 0.55


def test_reference_similarity_line_proximity_contributes():
    task = _make_task(line=5)
    ref = task.reference_findings[0]

    finding_on_line = ReviewFinding(
        title="Avoid eval",
        line=5,
        category="security",
        severity="critical",
        rationale="eval executes arbitrary code.",
    )
    finding_far = ReviewFinding(
        title="Avoid eval",
        line=50,
        category="security",
        severity="critical",
        rationale="eval executes arbitrary code.",
    )

    score_on = _reference_similarity(finding_on_line, ref)
    score_far = _reference_similarity(finding_far, ref)
    assert score_on > score_far


# ---------------------------------------------------------------------------
# _fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_same_finding_is_identical():
    finding = _good_finding()
    assert _fingerprint(finding) == _fingerprint(finding)


def test_fingerprint_different_line_differs():
    f1 = ReviewFinding(
        title="x",
        line=1,
        category="bug",
        severity="warning",
        rationale="r",
    )
    f2 = ReviewFinding(
        title="x",
        line=2,
        category="bug",
        severity="warning",
        rationale="r",
    )
    assert _fingerprint(f1) != _fingerprint(f2)


def test_fingerprint_none_line_handled():
    finding = ReviewFinding(
        title="test",
        line=None,
        category="bug",
        severity="warning",
        rationale="test rationale",
    )
    fp = _fingerprint(finding)
    assert fp  # should not raise and should be non-empty


# ---------------------------------------------------------------------------
# _grade_patch_score
# ---------------------------------------------------------------------------

def test_grade_patch_score_zero_for_no_code():
    task = get_task_by_id("py-review-easy")
    assert _grade_patch_score(task, None) == 0.0
    assert _grade_patch_score(task, "") == 0.0


def test_grade_patch_score_positive_when_eval_removed():
    task = get_task_by_id("py-review-easy")
    fixed = (
        "import json\n\n"
        "def load_settings(cfg):\n"
        "    return json.loads(cfg)\n\n"
        "def compute_average(total, count=1):\n"
        "    return total / count\n"
    )
    score = _grade_patch_score(task, fixed)
    assert score > 0.0


def test_grade_patch_score_zero_when_no_detectable_rules():
    # A task with only a non-detectable rule should return 0 even with patched code
    task = _make_task(rule_id="division-by-zero-default")
    score = _grade_patch_score(task, "def f(x):\n    return x\n")
    assert score == 0.0


# ---------------------------------------------------------------------------
# evaluate_submission – core grading flow
# ---------------------------------------------------------------------------

def test_evaluate_submission_matches_correct_finding():
    task = _make_task()
    grade = evaluate_submission(task=task, findings=[_good_finding()])

    assert grade.evaluation.matched_findings == 1
    assert "avoid-eval" in grade.newly_matched_ids


def test_evaluate_submission_no_findings_score_zero():
    task = _make_task()
    grade = evaluate_submission(task=task, findings=[])

    assert grade.evaluation.score == 0.0
    assert grade.evaluation.matched_findings == 0


def test_evaluate_submission_false_positive_counted():
    task = _make_task()
    junk = ReviewFinding(
        title="Made-up issue",
        line=99,
        category="style",
        severity="info",
        rationale="not real",
    )
    grade = evaluate_submission(task=task, findings=[junk])

    assert grade.false_positives >= 1
    assert grade.evaluation.false_positives >= 1


def test_evaluate_submission_duplicate_finding_counted():
    task = _make_task()
    finding = _good_finding()
    grade = evaluate_submission(
        task=task,
        findings=[finding, finding],
        prior_fingerprints=set(),
    )

    assert grade.duplicate_findings >= 1


def test_evaluate_submission_prior_matched_ids_respected():
    task = _make_task()
    # Pre-seed the matched set so the finding is already counted
    grade = evaluate_submission(
        task=task,
        findings=[_good_finding()],
        prior_matched_ids={"avoid-eval"},
    )

    # Should not count it again as a NEW match
    assert "avoid-eval" not in grade.newly_matched_ids


def test_evaluate_submission_use_existing_matches_skips_loop():
    task = _make_task()
    grade = evaluate_submission(
        task=task,
        findings=[_good_finding()],
        prior_matched_ids={"avoid-eval"},
        use_existing_matches=True,
    )

    # With use_existing_matches=True, the loop is skipped; matched_findings
    # reflects whatever is in prior_matched_ids
    assert grade.evaluation.matched_findings == 1
    assert grade.newly_matched_ids == []


def test_evaluate_submission_patch_score_propagated():
    task = get_task_by_id("py-review-easy")
    fixed = (
        "import json\n\n"
        "def load_settings(cfg):\n"
        "    return json.loads(cfg)\n\n"
        "def compute_average(total, count=1):\n"
        "    return total / count\n"
    )
    grade = evaluate_submission(task=task, findings=[], patched_code=fixed)

    assert grade.patch_score > 0.0
    assert grade.evaluation.patch_score > 0.0


def test_evaluate_submission_force_patch_score():
    task = _make_task()
    grade = evaluate_submission(
        task=task,
        findings=[],
        force_patch_score=0.75,
    )

    assert grade.patch_score == 0.75


def test_evaluate_submission_score_penalised_by_false_positives():
    task = _make_task(success_threshold=0.5)

    # One good match
    findings = [_good_finding()]
    # Add many junk findings to rack up false positives
    for i in range(5):
        findings.append(
            ReviewFinding(
                title=f"Fake finding {i}",
                line=i + 10,
                category="style",
                severity="info",
                rationale="not real",
            )
        )
    grade = evaluate_submission(task=task, findings=findings)

    # Penalty should reduce score below weighted recall
    assert grade.evaluation.score <= grade.evaluation.weighted_recall


def test_evaluate_submission_score_capped_at_one():
    task = _make_task()
    grade = evaluate_submission(task=task, findings=[_good_finding()])

    assert grade.evaluation.score <= 1.0


def test_evaluate_submission_passed_when_above_threshold():
    task = _make_task(success_threshold=0.5)
    grade = evaluate_submission(task=task, findings=[_good_finding()])

    assert grade.evaluation.passed is True


def test_evaluate_submission_failed_when_below_threshold():
    task = _make_task(success_threshold=0.99)
    grade = evaluate_submission(task=task, findings=[])

    assert grade.evaluation.passed is False


# ---------------------------------------------------------------------------
# Multi-finding task (medium task)
# ---------------------------------------------------------------------------

def test_evaluate_medium_task_partial_match():
    task = get_task_by_id("py-review-medium")

    findings = [
        ReviewFinding(
            title="Mutable default argument leaks state",
            line=1,
            category="bug",
            severity="warning",
            rationale="The default list is shared across calls.",
            recommendation="Use None as default.",
        )
    ]
    grade = evaluate_submission(task=task, findings=findings)

    assert grade.evaluation.matched_findings >= 1
    assert grade.evaluation.matched_findings < grade.evaluation.total_findings


# ---------------------------------------------------------------------------
# TaskSpec.max_steps property
# ---------------------------------------------------------------------------

def test_task_spec_max_steps_property():
    task = get_task_by_id("py-review-easy")

    assert task.max_steps == 4


# ---------------------------------------------------------------------------
# Compatibility shim modules
# ---------------------------------------------------------------------------

def test_code_review_env_environment_shim_imports():
    from server.code_review_env_environment import CodeReviewEnvironment

    assert CodeReviewEnvironment is not None


def test_python_env_environment_shim_imports():
    from server.python_env_environment import PythonEnvironment as ShimEnv

    assert ShimEnv is not None
