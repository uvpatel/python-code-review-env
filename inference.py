"""Baseline inference script for the Python code-review environment."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from client import PythonEnv
from models import PythonReviewAction, ReviewFinding


API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
DOCKER_IMAGE = os.getenv("PYTHON_ENV_IMAGE", "python_env-env:latest")
MAX_STEPS = int(os.getenv("MAX_STEPS", "3"))

SYSTEM_PROMPT = """You are a precise Python code reviewer.
Return strict JSON with this shape:
{
  "findings": [
    {
      "title": "short title",
      "line": 1,
      "category": "bug|security|style|performance|maintainability",
      "severity": "critical|warning|info",
      "rationale": "why it matters",
      "recommendation": "how to fix it"
    }
  ],
  "patched_code": null
}

Only report real issues visible in the code. Prefer high precision over quantity.
"""


def _build_prompt(observation) -> str:
    return (
        f"Task ID: {observation.task.task_id}\n"
        f"Difficulty: {observation.task.difficulty}\n"
        f"Objective: {observation.task.objective}\n"
        f"Attempts remaining: {observation.attempts_remaining}\n"
        f"Previous feedback: {observation.feedback}\n\n"
        "Code to review:\n"
        "```python\n"
        f"{observation.task.code}\n"
        "```"
    )


def _parse_response(content: str) -> Dict[str, Any]:
    try:
        data = json.loads(content)
        findings = data.get("findings", [])
        if not isinstance(findings, list):
            findings = []
        return {"findings": findings, "patched_code": data.get("patched_code")}
    except json.JSONDecodeError:
        return {"findings": [], "patched_code": None}


def _completion(client: OpenAI, prompt: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=800,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return _parse_response(content)


def _to_action(payload: Dict[str, Any], finalize: bool) -> PythonReviewAction:
    findings = []
    for item in payload.get("findings", []):
        try:
            findings.append(ReviewFinding(**item))
        except Exception:
            continue
    return PythonReviewAction(
        operation="finalize" if finalize else "submit_findings",
        findings=findings,
        patched_code=payload.get("patched_code"),
    )


def _make_env() -> PythonEnv:
    if ENV_BASE_URL:
        return PythonEnv(base_url=ENV_BASE_URL)
    return PythonEnv.from_docker_image(DOCKER_IMAGE)


def main() -> None:
    if not API_KEY:
        raise RuntimeError("Set HF_TOKEN or OPENAI_API_KEY before running inference.py")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = _make_env()
    episode_results: List[Dict[str, Any]] = []

    try:
        for index in range(3):
            result = env.reset()
            observation = result.observation
            print(f"Task {index + 1}: {observation.task.task_id} ({observation.task.difficulty})")
            for step in range(1, MAX_STEPS + 1):
                payload = _completion(client, _build_prompt(observation))
                action = _to_action(payload, finalize=step == MAX_STEPS)
                result = env.step(action)
                observation = result.observation
                print(
                    f"  step={step} score={observation.score:.2f} "
                    f"reward={(result.reward or 0.0):.2f} done={result.done}"
                )
                if result.done:
                    break
            episode_results.append(
                {
                    "task_id": observation.task.task_id,
                    "difficulty": observation.task.difficulty,
                    "score": observation.score,
                    "passed": observation.evaluation.passed,
                }
            )
    finally:
        env.close()

    mean_score = sum(item["score"] for item in episode_results) / len(episode_results)
    print(json.dumps({"results": episode_results, "mean_score": mean_score}, indent=2))


if __name__ == "__main__":
    main()
