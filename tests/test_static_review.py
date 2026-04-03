"""Tests for server/static_review.py – analyze_python_code and build_direct_review_response."""

import pytest

from server.static_review import analyze_python_code, build_direct_review_response


# ---------------------------------------------------------------------------
# analyze_python_code – basic rule detection
# ---------------------------------------------------------------------------

def test_detects_eval():
    issues = analyze_python_code("def f(x):\n    return eval(x)\n")

    rule_ids = [i.rule_id for i in issues]
    assert "avoid-eval" in rule_ids


def test_detects_exec():
    issues = analyze_python_code("def f(x):\n    exec(x)\n")

    rule_ids = [i.rule_id for i in issues]
    assert "avoid-exec" in rule_ids


def test_detects_shell_true():
    code = (
        "import subprocess\n"
        "def run(cmd):\n"
        "    subprocess.run(cmd, shell=True)\n"
    )
    issues = analyze_python_code(code)

    rule_ids = [i.rule_id for i in issues]
    assert "shell-true-command-injection" in rule_ids


def test_detects_check_output_shell_true():
    code = (
        "import subprocess\n"
        "def run(cmd):\n"
        "    subprocess.check_output(cmd, shell=True)\n"
    )
    issues = analyze_python_code(code)

    rule_ids = [i.rule_id for i in issues]
    assert "shell-true-command-injection" in rule_ids


def test_detects_mutable_default_list():
    issues = analyze_python_code("def f(items=[]):\n    return items\n")

    rule_ids = [i.rule_id for i in issues]
    assert "mutable-default-list" in rule_ids


def test_detects_mutable_default_dict():
    issues = analyze_python_code("def f(cfg={}):\n    return cfg\n")

    rule_ids = [i.rule_id for i in issues]
    assert "mutable-default-list" in rule_ids


def test_detects_mutable_default_set():
    issues = analyze_python_code("def f(s=set()):\n    return s\n")

    # set() is a Call node, not a Set literal – should not trigger
    rule_ids = [i.rule_id for i in issues]
    assert "mutable-default-list" not in rule_ids


def test_detects_bare_except():
    code = "try:\n    pass\nexcept:\n    pass\n"
    issues = analyze_python_code(code)

    rule_ids = [i.rule_id for i in issues]
    assert "bare-except" in rule_ids


def test_specific_except_not_flagged():
    code = "try:\n    pass\nexcept ValueError:\n    pass\n"
    issues = analyze_python_code(code)

    rule_ids = [i.rule_id for i in issues]
    assert "bare-except" not in rule_ids


def test_detects_quadratic_membership_check_in_loop():
    code = (
        "def dedup(rows):\n"
        "    seen = []\n"
        "    for row in rows:\n"
        "        if row not in seen:\n"
        "            seen.append(row)\n"
        "    return seen\n"
    )
    issues = analyze_python_code(code)

    rule_ids = [i.rule_id for i in issues]
    assert "quadratic-membership-check" in rule_ids


def test_membership_check_not_flagged_outside_loop():
    code = "def f(x, items):\n    return x in items\n"
    issues = analyze_python_code(code)

    rule_ids = [i.rule_id for i in issues]
    assert "quadratic-membership-check" not in rule_ids


def test_detects_print_statement():
    issues = analyze_python_code("def f():\n    print('hello')\n")

    rule_ids = [i.rule_id for i in issues]
    assert "print-statement" in rule_ids


def test_clean_code_returns_no_issues():
    code = "def add(a, b):\n    return a + b\n"
    issues = analyze_python_code(code)

    assert issues == []


# ---------------------------------------------------------------------------
# Edge cases – empty input and syntax error
# ---------------------------------------------------------------------------

def test_empty_code_returns_empty_input_issue():
    issues = analyze_python_code("")

    assert len(issues) == 1
    assert issues[0].rule_id == "empty-input"


def test_whitespace_only_code_returns_empty_input_issue():
    issues = analyze_python_code("   \n\t  ")

    assert len(issues) == 1
    assert issues[0].rule_id == "empty-input"


def test_syntax_error_returns_syntax_error_issue():
    issues = analyze_python_code("def f(\n    # missing closing paren")

    assert len(issues) == 1
    assert issues[0].rule_id == "syntax-error"
    assert issues[0].severity == "critical"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_duplicate_issues_are_deduplicated():
    # Two calls to eval on the same line would normally produce two findings
    # for the same (rule_id, line, category) – only one should survive.
    code = "def f(x):\n    y = eval(x); z = eval(x)\n"
    issues = analyze_python_code(code)

    eval_issues = [i for i in issues if i.rule_id == "avoid-eval"]
    assert len(eval_issues) == 1


# ---------------------------------------------------------------------------
# Multiple issues in same snippet
# ---------------------------------------------------------------------------

def test_multiple_issues_detected_together():
    code = (
        "def load(cfg_text, items=[]):\n"
        "    settings = eval(cfg_text)\n"
        "    return settings\n"
    )
    issues = analyze_python_code(code)

    rule_ids = {i.rule_id for i in issues}
    assert "avoid-eval" in rule_ids
    assert "mutable-default-list" in rule_ids


# ---------------------------------------------------------------------------
# build_direct_review_response
# ---------------------------------------------------------------------------

def test_build_response_score_one_for_clean_code():
    result = build_direct_review_response("def add(a, b):\n    return a + b\n")

    assert result.score == 1.0
    assert result.issues == []
    assert result.improved_code is None


def test_build_response_score_reduced_for_critical_issue():
    result = build_direct_review_response("def f(x):\n    return eval(x)\n")

    assert result.score < 1.0
    assert result.issues


def test_build_response_summary_mentions_issue_counts():
    result = build_direct_review_response("def f(x):\n    return eval(x)\n")

    assert "critical" in result.summary.lower()


def test_build_response_summary_no_issues():
    result = build_direct_review_response("x = 1\n")

    assert "no obvious issues" in result.summary.lower()


def test_build_response_summary_includes_context():
    result = build_direct_review_response(
        "def f(x):\n    return eval(x)\n",
        context="user input handler",
    )

    assert "user input handler" in result.summary


def test_build_response_improved_code_contains_suggestions():
    result = build_direct_review_response("def f(x):\n    return eval(x)\n")

    assert result.improved_code is not None
    assert "Suggested review directions" in result.improved_code


def test_build_response_score_reduced_for_info_severity():
    # print() generates an info-severity issue (rule_id="print-statement")
    result = build_direct_review_response("def f():\n    print('hello')\n")

    assert result.score < 1.0
    assert any(i.severity == "info" for i in result.issues)


def test_build_response_score_clamped_to_zero_for_many_issues():
    code = (
        "def f(x, data=[]):\n"
        "    eval(x)\n"
        "    exec(x)\n"
        "    import subprocess\n"
        "    subprocess.run(x, shell=True)\n"
    )
    result = build_direct_review_response(code)

    assert result.score >= 0.0
    assert result.score <= 1.0


def test_call_via_deep_attribute_chain_is_detected():
    # Covers _attribute_prefix with nested Attribute nodes (a.b.run(shell=True))
    code = (
        "def f(cmd):\n"
        "    import a\n"
        "    a.b.run(cmd, shell=True)\n"
    )
    result = build_direct_review_response(code)

    assert any(i.rule_id == "shell-true-command-injection" for i in result.issues)


def test_call_via_subscript_does_not_crash():
    # Covers the fallback '' return in _call_name (non-Name/Attribute func)
    code = "funcs = [print]\nfuncs[0]('hello')\n"
    result = build_direct_review_response(code)

    assert result is not None


def test_call_with_complex_receiver_does_not_crash():
    # Covers _attribute_prefix fallback (return "") when receiver is not Name/Attribute
    # e.g. (get_runner()).run(cmd, shell=True) – func value is a Call, not a Name
    code = "def f(cmd):\n    get_runner().run(cmd, shell=True)\n"
    result = build_direct_review_response(code)

    # The rule may or may not fire depending on whether the prefix can be resolved,
    # but the analyser must not raise.
    assert result is not None
