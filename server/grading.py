"""Deterministic graders for benchmark tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

try:
    from models import ReviewFinding, TaskEvaluation
    from server.static_review import analyze_python_code
    from server.task_bank import ReferenceFinding, TaskSpec
except ModuleNotFoundError:  # pragma: no cover
    from ..models import ReviewFinding, TaskEvaluation
    from .static_review import analyze_python_code
    from .task_bank import ReferenceFinding, TaskSpec


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass
class SubmissionGrade:
    evaluation: TaskEvaluation
    newly_matched_ids: List[str]
    accepted_fingerprints: Set[str]
    false_positives: int
    duplicate_findings: int
    patch_score: float


def evaluate_submission(
    task: TaskSpec,
    findings: Sequence[ReviewFinding],
    patched_code: Optional[str] = None,
    prior_matched_ids: Optional[Set[str]] = None,
    prior_fingerprints: Optional[Set[str]] = None,
    force_patch_score: Optional[float] = None,
    use_existing_matches: bool = False,
    false_positives: int = 0,
    duplicate_findings: int = 0,
) -> SubmissionGrade:
    matched_ids: Set[str] = set(prior_matched_ids or [])
    accepted_fingerprints: Set[str] = set()
    seen_fingerprints: Set[str] = set(prior_fingerprints or set())
    new_matches: List[str] = []
    current_false_positives = false_positives
    current_duplicates = duplicate_findings

    if not use_existing_matches:
        for finding in findings:
            fingerprint = _fingerprint(finding)
            if fingerprint in seen_fingerprints:
                current_duplicates += 1
                continue
            seen_fingerprints.add(fingerprint)
            accepted_fingerprints.add(fingerprint)
            reference = _match_finding(finding, task.reference_findings, matched_ids)
            if reference is None:
                current_false_positives += 1
                continue
            matched_ids.add(reference.finding_id)
            new_matches.append(reference.finding_id)

    patch_score = (
        force_patch_score
        if force_patch_score is not None
        else _grade_patch_score(task, patched_code)
    )
    evaluation = _build_evaluation(
        task=task,
        matched_ids=matched_ids,
        false_positives=current_false_positives,
        duplicate_findings=current_duplicates,
        patch_score=patch_score,
    )
    return SubmissionGrade(
        evaluation=evaluation,
        newly_matched_ids=new_matches,
        accepted_fingerprints=accepted_fingerprints,
        false_positives=current_false_positives - false_positives,
        duplicate_findings=current_duplicates - duplicate_findings,
        patch_score=patch_score,
    )


def _build_evaluation(
    task: TaskSpec,
    matched_ids: Set[str],
    false_positives: int,
    duplicate_findings: int,
    patch_score: float,
) -> TaskEvaluation:
    total_weight = sum(reference.weight for reference in task.reference_findings) or 1.0
    matched_weight = sum(
        reference.weight
        for reference in task.reference_findings
        if reference.finding_id in matched_ids
    )
    weighted_recall = max(0.0, min(1.0, matched_weight / total_weight))
    penalty = min(0.35, false_positives * 0.08 + duplicate_findings * 0.03)
    score = max(0.0, min(1.0, weighted_recall - penalty))
    return TaskEvaluation(
        matched_reference_ids=sorted(matched_ids),
        matched_findings=len(matched_ids),
        total_findings=len(task.reference_findings),
        false_positives=false_positives,
        duplicate_findings=duplicate_findings,
        weighted_recall=weighted_recall,
        patch_score=patch_score,
        score=score,
        passed=score >= task.success_threshold,
    )


def _match_finding(
    finding: ReviewFinding,
    references: Sequence[ReferenceFinding],
    matched_ids: Set[str],
) -> Optional[ReferenceFinding]:
    best_reference: Optional[ReferenceFinding] = None
    best_score = 0.0
    for reference in references:
        if reference.finding_id in matched_ids:
            continue
        score = _reference_similarity(finding, reference)
        if score > best_score:
            best_score = score
            best_reference = reference
    if best_score >= 0.55:
        return best_reference
    return None


def _reference_similarity(finding: ReviewFinding, reference: ReferenceFinding) -> float:
    score = 0.0
    if finding.category == reference.category:
        score += 0.3
    if finding.severity == reference.severity:
        score += 0.15
    if finding.line is not None and abs(finding.line - reference.line) <= 1:
        score += 0.25
    text_tokens = _tokens(
        " ".join(
            part
            for part in (
                finding.title,
                finding.rationale,
                finding.recommendation or "",
                finding.rule_id or "",
            )
            if part
        )
    )
    reference_phrases = [reference.title, reference.rule_id, *reference.aliases]
    best_phrase_overlap = max(_phrase_overlap(text_tokens, phrase) for phrase in reference_phrases)
    score += best_phrase_overlap * 0.35
    return min(score, 1.0)


def _grade_patch_score(task: TaskSpec, patched_code: Optional[str]) -> float:
    if not patched_code:
        return 0.0
    issues = analyze_python_code(patched_code)
    remaining_rule_ids = {issue.rule_id for issue in issues if issue.rule_id}
    detectable_rule_ids = {
        reference.rule_id
        for reference in task.reference_findings
        if reference.rule_id
        and reference.rule_id
        in {
            "avoid-eval",
            "mutable-default-list",
            "quadratic-membership-check",
            "bare-except",
            "shell-true-command-injection",
        }
    }
    if not detectable_rule_ids:
        return 0.0
    fixed = len(detectable_rule_ids - remaining_rule_ids)
    return max(0.0, min(1.0, fixed / len(detectable_rule_ids)))


def _phrase_overlap(tokens: Set[str], phrase: str) -> float:
    phrase_tokens = _tokens(phrase)
    if not phrase_tokens:
        return 0.0
    overlap = len(tokens & phrase_tokens)
    if phrase_tokens.issubset(tokens):
        return 1.0
    return overlap / len(phrase_tokens)


def _tokens(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_]+", text.lower())
        if token not in STOPWORDS
    }


def _fingerprint(finding: ReviewFinding) -> str:
    tokens = sorted(_tokens(f"{finding.title} {finding.rationale} {finding.recommendation or ''}"))
    return "|".join(
        [
            finding.category,
            finding.severity,
            str(finding.line or 0),
            finding.rule_id or "",
            " ".join(tokens),
        ]
    )
