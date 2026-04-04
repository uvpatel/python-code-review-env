from examples.python_review_examples import EXAMPLE_SNIPPETS, EXPECTED_RULE_IDS
from server.static_review import analyze_python_code, build_direct_review_response


def test_example_snippets_produce_expected_rule_ids():
    for name, code in EXAMPLE_SNIPPETS.items():
        issues = analyze_python_code(code)
        rule_ids = {issue.rule_id for issue in issues if issue.rule_id}
        assert EXPECTED_RULE_IDS[name].issubset(rule_ids)


def test_clean_example_scores_well():
    response = build_direct_review_response(EXAMPLE_SNIPPETS["clean_function"])

    assert response.issues == []
    assert response.score == 1.0


def test_shell_injection_example_is_critical():
    response = build_direct_review_response(EXAMPLE_SNIPPETS["shell_injection"])

    assert any(issue.severity == "critical" for issue in response.issues)
    assert any(issue.rule_id == "shell-true-command-injection" for issue in response.issues)
