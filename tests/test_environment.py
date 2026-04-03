from models import PythonEnvConfig, PythonReviewAction, ReviewFinding
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


# ---------------------------------------------------------------------------
# Hint operations
# ---------------------------------------------------------------------------

def test_request_hint_returns_hint_text():
    env = PythonEnvironment()
    env.reset()

    obs = env.step(PythonReviewAction(operation="request_hint"))

    assert "Hint 1" in obs.feedback
    assert obs.hints_used == 1


def test_request_hint_applies_penalty():
    env = PythonEnvironment()
    env.reset()

    obs = env.step(PythonReviewAction(operation="request_hint"))

    assert obs.reward < 0


def test_requesting_all_hints_then_exhausted():
    env = PythonEnvironment()
    env.reset()

    # easy task has 2 hints
    env.step(PythonReviewAction(operation="request_hint"))
    env.step(PythonReviewAction(operation="request_hint"))
    obs = env.step(PythonReviewAction(operation="request_hint"))

    assert "No hints remaining" in obs.feedback


# ---------------------------------------------------------------------------
# Duplicate and false-positive tracking
# ---------------------------------------------------------------------------

def test_duplicate_finding_is_penalised():
    env = PythonEnvironment()
    env.reset()

    finding = ReviewFinding(
        title="Avoid eval on untrusted input",
        line=2,
        category="security",
        severity="critical",
        rationale="eval can execute attacker-controlled code.",
        recommendation="Use json.loads instead.",
    )
    env.step(PythonReviewAction(findings=[finding]))
    obs2 = env.step(PythonReviewAction(findings=[finding]))

    assert obs2.evaluation.duplicate_findings >= 1


def test_false_positive_is_penalised():
    env = PythonEnvironment()
    env.reset()

    finding = ReviewFinding(
        title="Completely made up issue",
        line=99,
        category="bug",
        severity="warning",
        rationale="Does not correspond to anything real.",
    )
    obs = env.step(PythonReviewAction(findings=[finding]))

    assert obs.evaluation.false_positives >= 1


# ---------------------------------------------------------------------------
# Max-steps auto-close
# ---------------------------------------------------------------------------

def test_max_steps_auto_closes_episode():
    env = PythonEnvironment(config=PythonEnvConfig(max_steps_per_task=2))
    env.reset()

    env.step(PythonReviewAction(operation="submit_findings"))
    obs = env.step(PythonReviewAction(operation="submit_findings"))

    assert obs.done is True
    assert "Maximum steps" in obs.feedback


# ---------------------------------------------------------------------------
# Already-done episode
# ---------------------------------------------------------------------------

def test_step_after_done_returns_negative_reward():
    env = PythonEnvironment()
    env.reset()

    env.step(PythonReviewAction(operation="finalize"))
    obs = env.step(PythonReviewAction(operation="submit_findings"))

    assert obs.reward < 0
    assert "already completed" in obs.feedback.lower()


# ---------------------------------------------------------------------------
# Task wrap-around
# ---------------------------------------------------------------------------

def test_task_order_wraps_around():
    env = PythonEnvironment()

    for _ in range(3):
        env.reset()

    fourth = env.reset()
    assert fourth.task.task_id == "py-review-easy"


# ---------------------------------------------------------------------------
# Non-passing finalize
# ---------------------------------------------------------------------------

def test_finalize_without_matches_does_not_pass():
    env = PythonEnvironment()
    env.reset()

    obs = env.step(PythonReviewAction(operation="finalize", findings=[]))

    assert obs.done is True
    assert obs.evaluation.passed is False


# ---------------------------------------------------------------------------
# Step without prior reset auto-resets
# ---------------------------------------------------------------------------

def test_step_without_reset_auto_resets():
    env = PythonEnvironment()

    obs = env.step(PythonReviewAction(operation="submit_findings"))

    assert obs.task is not None
    assert obs.task.task_id is not None


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------

def test_history_grows_after_reset():
    env = PythonEnvironment()
    env.reset()

    assert len(env.get_history()) == 1


def test_clear_history():
    env = PythonEnvironment()
    env.reset()
    env.clear_history()

    assert env.get_history() == []


def test_history_max_entries_respected():
    env = PythonEnvironment(config=PythonEnvConfig(max_history_entries=2))

    for _ in range(5):
        env.reset()

    assert len(env.get_history()) <= 2


# ---------------------------------------------------------------------------
# update_config
# ---------------------------------------------------------------------------

def test_update_config_populates_task_order_when_empty():
    env = PythonEnvironment()
    # PythonEnvConfig with empty task_order triggers the fallback branch
    config = PythonEnvConfig(task_order=[], max_steps_per_task=4)
    env.update_config(config)

    assert len(env.config.task_order) == 3


def test_state_property_returns_state_object():
    env = PythonEnvironment()
    env.reset()

    from openenv.core.env_server.types import State
    assert isinstance(env.state, State)


def test_update_config_changes_max_steps():
    env = PythonEnvironment()
    new_config = PythonEnvConfig(max_steps_per_task=7)
    env.update_config(new_config)

    assert env.config.max_steps_per_task == 7
    obs = env.reset()
    assert obs.task.max_steps == 7


# ---------------------------------------------------------------------------
# Patched code bonus
# ---------------------------------------------------------------------------

def test_patch_with_fixed_eval_gives_patch_score():
    env = PythonEnvironment()
    env.reset()

    patched = (
        "import json\n\n"
        "def load_settings(config_text):\n"
        "    settings = json.loads(config_text)\n"
        "    return settings\n\n"
        "def compute_average(total, count=1):\n"
        "    return total / count\n"
    )
    obs = env.step(
        PythonReviewAction(
            operation="submit_findings",
            findings=[],
            patched_code=patched,
        )
    )

    assert obs.evaluation.patch_score > 0


# ---------------------------------------------------------------------------
# Reviewer note
# ---------------------------------------------------------------------------

def test_reviewer_note_is_recorded_in_feedback():
    env = PythonEnvironment()
    env.reset()

    obs = env.step(
        PythonReviewAction(
            operation="submit_findings",
            findings=[],
            note="checking for eval usage",
        )
    )

    assert "checking for eval usage" in obs.feedback


# ---------------------------------------------------------------------------
# medium and hard task finalise
# ---------------------------------------------------------------------------

def test_finalize_medium_task_with_correct_findings():
    env = PythonEnvironment()
    env.reset()  # easy
    env.reset()  # medium

    obs = env.step(
        PythonReviewAction(
            operation="finalize",
            findings=[
                ReviewFinding(
                    title="Mutable default argument leaks state between calls",
                    line=1,
                    category="bug",
                    severity="warning",
                    rationale="The default list is shared across calls.",
                    recommendation="Use None and create the list inside the function.",
                ),
                ReviewFinding(
                    title="List membership inside loop is quadratic",
                    line=5,
                    category="performance",
                    severity="warning",
                    rationale="Repeated membership checks degrade to O(n^2).",
                    recommendation="Use a set for O(1) membership checks.",
                ),
                ReviewFinding(
                    title="Bare except swallows all errors",
                    line=12,
                    category="maintainability",
                    severity="warning",
                    rationale="Bare except hides all exceptions including KeyboardInterrupt.",
                    recommendation="Catch specific exceptions and log the failure.",
                ),
            ],
        )
    )

    assert obs.done is True
    assert obs.score >= 0.75

