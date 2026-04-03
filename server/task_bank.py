"""Benchmark task definitions for the Python code-review environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

try:
    from models import Severity, Category, Difficulty, TaskDescriptor
except ModuleNotFoundError:  # pragma: no cover
    from ..models import Severity, Category, Difficulty, TaskDescriptor


@dataclass(frozen=True)
class ReferenceFinding:
    finding_id: str
    title: str
    category: Category
    severity: Severity
    line: int
    aliases: Sequence[str]
    recommendation: str
    weight: float = 1.0
    rule_id: str = ""


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    difficulty: Difficulty
    title: str
    objective: str
    code: str
    hints: Sequence[str] = field(default_factory=tuple)
    reference_findings: Sequence[ReferenceFinding] = field(default_factory=tuple)
    success_threshold: float = 0.75

    @property
    def max_steps(self) -> int:
        return 4

    def to_descriptor(self, max_steps: int) -> TaskDescriptor:
        return TaskDescriptor(
            task_id=self.task_id,
            difficulty=self.difficulty,
            title=self.title,
            objective=self.objective,
            code=self.code,
            max_steps=max_steps,
            success_threshold=self.success_threshold,
        )


TASKS: List[TaskSpec] = [
    TaskSpec(
        task_id="py-review-easy",
        difficulty="easy",
        title="Unsafe Config Loader",
        objective=(
            "Review a tiny helper module and identify the most important correctness "
            "and security problems before it ships."
        ),
        code="\n".join(
            [
                "def load_settings(config_text):",
                "    settings = eval(config_text)",
                "    return settings",
                "",
                "def compute_average(total, count=0):",
                "    return total / count",
            ]
        ),
        hints=(
            "One issue is a direct code-execution risk.",
            "A default parameter makes the arithmetic helper crash on common input.",
        ),
        reference_findings=(
            ReferenceFinding(
                finding_id="avoid-eval",
                title="Avoid eval on untrusted configuration data",
                category="security",
                severity="critical",
                line=2,
                aliases=("eval execution", "code injection", "unsafe eval"),
                recommendation="Parse structured config with json.loads or ast.literal_eval.",
                weight=1.4,
                rule_id="avoid-eval",
            ),
            ReferenceFinding(
                finding_id="division-by-zero-default",
                title="Default count of zero causes a division by zero",
                category="bug",
                severity="warning",
                line=5,
                aliases=("divide by zero", "count default zero", "zero divisor"),
                recommendation="Validate count before dividing or pick a safe default.",
                weight=1.0,
                rule_id="division-by-zero-default",
            ),
        ),
        success_threshold=0.8,
    ),
    TaskSpec(
        task_id="py-review-medium",
        difficulty="medium",
        title="Batch Email Cleanup Utility",
        objective=(
            "Review a data-cleaning helper used in a backend job. Flag bugs, hidden "
            "operational risks, and obvious scalability issues."
        ),
        code="\n".join(
            [
                "def collect_unique_emails(rows=[]):",
                "    unique_emails = []",
                "    for row in rows:",
                '        email = row["email"].lower()',
                "        if email not in unique_emails:",
                "            unique_emails.append(email)",
                "    return unique_emails",
                "",
                "def publish_report(report):",
                "    try:",
                '        print(report["summary"])',
                "    except:",
                "        pass",
            ]
        ),
        hints=(
            "One bug comes from state leaking across calls.",
            "The membership check becomes expensive as the dataset grows.",
            "The exception handling hides failures that operators would need to see.",
        ),
        reference_findings=(
            ReferenceFinding(
                finding_id="mutable-default-list",
                title="Mutable default argument leaks state between calls",
                category="bug",
                severity="warning",
                line=1,
                aliases=(
                    "mutable default argument",
                    "default list reused",
                    "shared default state",
                ),
                recommendation="Use None as the default and create the list inside the function.",
                weight=1.2,
                rule_id="mutable-default-list",
            ),
            ReferenceFinding(
                finding_id="quadratic-membership-check",
                title="List membership inside the loop makes deduplication quadratic",
                category="performance",
                severity="warning",
                line=5,
                aliases=(
                    "o n squared",
                    "quadratic lookup",
                    "set membership",
                ),
                recommendation="Track seen emails in a set for O(1) membership checks.",
                weight=1.0,
                rule_id="quadratic-membership-check",
            ),
            ReferenceFinding(
                finding_id="bare-except",
                title="Bare except swallows all errors and hides broken reports",
                category="maintainability",
                severity="warning",
                line=12,
                aliases=("bare except", "swallow exception", "except pass"),
                recommendation="Catch specific exceptions and log the failure.",
                weight=1.1,
                rule_id="bare-except",
            ),
        ),
        success_threshold=0.75,
    ),
    TaskSpec(
        task_id="py-review-hard",
        difficulty="hard",
        title="Student Runner Sandbox",
        objective=(
            "Review a helper that executes student scripts. The code is headed toward a "
            "shared service, so correctness, security, and operational isolation all matter."
        ),
        code="\n".join(
            [
                "import subprocess",
                "from pathlib import Path",
                "",
                "CACHE = {}",
                "",
                "def run_student_script(script_path, user_input):",
                "    if script_path in CACHE:",
                "        return CACHE[script_path]",
                '    command = f"python {script_path} {user_input}"',
                "    output = subprocess.check_output(command, shell=True, text=True)",
                "    CACHE[script_path] = output",
                '    Path("latest_output.txt").write_text(output)',
                "    return output",
            ]
        ),
        hints=(
            "The process launch path is unsafe for untrusted input.",
            "The cache key ignores one of the function inputs.",
            "A fixed output file is risky in a concurrent or multi-user service.",
        ),
        reference_findings=(
            ReferenceFinding(
                finding_id="shell-true-command-injection",
                title="shell=True with interpolated input allows command injection",
                category="security",
                severity="critical",
                line=10,
                aliases=(
                    "shell true injection",
                    "command injection",
                    "unsafe subprocess shell",
                ),
                recommendation="Pass a list of arguments to subprocess without shell=True.",
                weight=1.5,
                rule_id="shell-true-command-injection",
            ),
            ReferenceFinding(
                finding_id="cache-key-misses-user-input",
                title="Cache key ignores user_input and can return stale output",
                category="bug",
                severity="warning",
                line=7,
                aliases=(
                    "cache key missing input",
                    "stale cache result",
                    "cache ignores user input",
                ),
                recommendation="Use both script_path and user_input in the cache key.",
                weight=1.2,
                rule_id="cache-key-misses-user-input",
            ),
            ReferenceFinding(
                finding_id="fixed-output-file",
                title="Writing every run to one fixed file can overwrite concurrent jobs",
                category="maintainability",
                severity="warning",
                line=12,
                aliases=(
                    "fixed output file",
                    "shared latest_output file",
                    "overwrite concurrent output",
                ),
                recommendation="Write to a unique per-run path or avoid filesystem side effects.",
                weight=1.0,
                rule_id="fixed-output-file",
            ),
        ),
        success_threshold=0.75,
    ),
]


TASKS_BY_ID: Dict[str, TaskSpec] = {task.task_id: task for task in TASKS}


def get_task_descriptors(max_steps: int) -> List[TaskDescriptor]:
    return [task.to_descriptor(max_steps) for task in TASKS]


def get_task_by_id(task_id: str) -> TaskSpec:
    try:
        return TASKS_BY_ID[task_id]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown task_id: {task_id}") from exc
