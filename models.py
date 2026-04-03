# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the code review environment.
"""

from typing import List, Optional, Literal

from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation


Severity = Literal["critical", "warning", "info"]
Category = Literal["bug", "security", "style", "performance"]


class CodeReviewAction(Action):
    """Submit code for review."""

    code: str = Field(..., description="Source code to review")
    language: str = Field(default="python", description="Programming language")
    focus_areas: List[str] = Field(
        default_factory=lambda: ["bugs", "style", "security", "performance"],
        description="Areas to focus the review on",
    )
    context: Optional[str] = Field(
        default=None, description="Optional context describing the code purpose"
    )


class ReviewIssue(BaseModel):
    severity: Severity = Field(..., description="Severity of the issue")
    category: Category = Field(..., description="Category of the code issue")
    line: Optional[int] = Field(
        default=None, description="Line number where the issue was identified"
    )
    message: str = Field(..., description="Human-readable issue description")
    suggestion: Optional[str] = Field(
        default=None, description="Optional suggestion to fix the issue"
    )


class CodeReviewConfig(BaseModel):
    focus_areas: List[str] = Field(
        default_factory=lambda: ["bugs", "style", "security", "performance"],
        description="Areas the reviewer should prioritize",
    )
    model_name: str = Field(
        default="placeholder-llm", description="Model used for the review"
    )


class CodeReviewObservation(Observation):
    """Review result from the environment."""

    original_code: str = Field(default="", description="Code that was reviewed")
    language: str = Field(default="python", description="Language of the submitted code")
    issues: List[ReviewIssue] = Field(default_factory=list, description="Detected issues")
    summary: str = Field(default="", description="Summary of the review")
    score: float = Field(
        default=0.0,
        description="Quality score between 0 (worst) and 10 (best)",
        ge=0.0,
        le=10.0,
    )
    improved_code: Optional[str] = Field(
        default=None, description="Optional suggested refactoring"
    )
    review_time_ms: float = Field(default=0.0, description="Time spent reviewing code")
