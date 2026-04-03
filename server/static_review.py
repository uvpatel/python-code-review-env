"""Deterministic static-review helpers for arbitrary Python code."""

from __future__ import annotations

import ast
from typing import List, Optional

try:
    from models import DirectReviewResponse, ReviewFinding
except ModuleNotFoundError:  # pragma: no cover
    from ..models import DirectReviewResponse, ReviewFinding


class _StaticAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: List[ReviewFinding] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        for default in list(node.args.defaults):
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.issues.append(
                    ReviewFinding(
                        title="Mutable default argument",
                        line=getattr(default, "lineno", node.lineno),
                        category="bug",
                        severity="warning",
                        rationale=(
                            "Mutable defaults persist across calls and can leak state "
                            "between unrelated requests."
                        ),
                        recommendation="Use None as the default and create the object inside the function.",
                        rule_id="mutable-default-list",
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        func_name = self._call_name(node)
        if func_name in {"eval", "exec"}:
            self.issues.append(
                ReviewFinding(
                    title=f"Avoid {func_name} on untrusted input",
                    line=node.lineno,
                    category="security",
                    severity="critical",
                    rationale=(
                        f"{func_name} executes arbitrary code and is unsafe on "
                        "user-controlled input."
                    ),
                    recommendation="Use a safe parser or a whitelist-based evaluator.",
                    rule_id="avoid-eval" if func_name == "eval" else "avoid-exec",
                )
            )
        if func_name.endswith("check_output") or func_name.endswith("run"):
            for keyword in node.keywords:
                if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    self.issues.append(
                        ReviewFinding(
                            title="shell=True with dynamic input",
                            line=node.lineno,
                            category="security",
                            severity="critical",
                            rationale=(
                                "shell=True executes through the shell and can allow "
                                "command injection when the command string is interpolated."
                            ),
                            recommendation="Pass a list of arguments and keep shell=False.",
                            rule_id="shell-true-command-injection",
                        )
                    )
        if func_name == "print":
            self.issues.append(
                ReviewFinding(
                    title="Print statement in application logic",
                    line=node.lineno,
                    category="style",
                    severity="info",
                    rationale="Production services should prefer structured logging over print statements.",
                    recommendation="Use the logging module or return the value to the caller.",
                    rule_id="print-statement",
                )
            )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        if node.type is None:
            self.issues.append(
                ReviewFinding(
                    title="Bare except",
                    line=node.lineno,
                    category="maintainability",
                    severity="warning",
                    rationale="Bare except catches KeyboardInterrupt and other system-level exceptions.",
                    recommendation="Catch a specific exception and record the failure.",
                    rule_id="bare-except",
                )
            )
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        for child in ast.walk(node):
            if isinstance(child, ast.Compare) and any(
                isinstance(operator, (ast.In, ast.NotIn)) for operator in child.ops
            ):
                if isinstance(child.comparators[0], ast.Name):
                    self.issues.append(
                        ReviewFinding(
                            title="Potential quadratic membership check inside loop",
                            line=child.lineno,
                            category="performance",
                            severity="warning",
                            rationale=(
                                "Repeated membership checks against a list inside a loop "
                                "can degrade to quadratic runtime."
                            ),
                            recommendation="Use a set or dict for O(1) membership checks.",
                            rule_id="quadratic-membership-check",
                        )
                    )
                    break
        self.generic_visit(node)

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            prefix = _StaticAnalyzer._attribute_prefix(func.value)
            return f"{prefix}.{func.attr}" if prefix else func.attr
        return ""

    @staticmethod
    def _attribute_prefix(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = _StaticAnalyzer._attribute_prefix(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""


def analyze_python_code(code: str) -> List[ReviewFinding]:
    if not code.strip():
        return [
            ReviewFinding(
                title="No code provided",
                category="bug",
                severity="warning",
                rationale="The reviewer cannot inspect an empty submission.",
                recommendation="Provide Python source code.",
                rule_id="empty-input",
            )
        ]

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [
            ReviewFinding(
                title="Syntax error",
                line=exc.lineno,
                category="bug",
                severity="critical",
                rationale=exc.msg,
                recommendation="Fix the syntax error before running static review.",
                rule_id="syntax-error",
            )
        ]

    analyzer = _StaticAnalyzer()
    analyzer.visit(tree)
    return _deduplicate(analyzer.issues)


def build_direct_review_response(
    code: str, context: Optional[str] = None
) -> DirectReviewResponse:
    issues = analyze_python_code(code)
    weighted_penalty = 0.0
    for issue in issues:
        if issue.severity == "critical":
            weighted_penalty += 0.3
        elif issue.severity == "warning":
            weighted_penalty += 0.15
        else:
            weighted_penalty += 0.05

    score = max(0.0, min(1.0, 1.0 - weighted_penalty))
    summary = _build_summary(issues, context)
    improved_code = _suggest_improved_code(code, issues)
    return DirectReviewResponse(
        issues=issues,
        summary=summary,
        score=score,
        improved_code=improved_code,
    )


def _build_summary(issues: List[ReviewFinding], context: Optional[str]) -> str:
    if not issues:
        base = "No obvious issues were detected by the deterministic reviewer."
    else:
        critical = sum(1 for issue in issues if issue.severity == "critical")
        warnings = sum(1 for issue in issues if issue.severity == "warning")
        infos = sum(1 for issue in issues if issue.severity == "info")
        base = (
            f"Detected {len(issues)} issue(s): {critical} critical, "
            f"{warnings} warning, {infos} info."
        )
    if context:
        return f"{base} Context: {context}"
    return base


def _suggest_improved_code(code: str, issues: List[ReviewFinding]) -> Optional[str]:
    if not issues:
        return None
    suggestions = [issue.recommendation for issue in issues if issue.recommendation]
    comment = " | ".join(dict.fromkeys(suggestions))
    return f"{code.rstrip()}\n\n# Suggested review directions: {comment}"


def _deduplicate(findings: List[ReviewFinding]) -> List[ReviewFinding]:
    seen = set()
    unique: List[ReviewFinding] = []
    for finding in findings:
        key = (finding.rule_id, finding.line, finding.category)
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique
