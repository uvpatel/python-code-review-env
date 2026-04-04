"""Baseline inference script for the Python PR review OpenEnv."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import PythonEnv
from models import PythonReviewAction, ReviewFinding


API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
DOCKER_IMAGE = os.getenv("PYTHON_ENV_IMAGE", "python_env-env:latest")
REPORT_PATH = Path(os.getenv("INFERENCE_REPORT_PATH", "inference_results.json"))
TEMPERATURE = 0.0
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "700"))

SYSTEM_PROMPT = """You are reviewing a Python pull request.
Return strict JSON only with one action per response.

Schema:
{
  "operation": "read_file" | "add_finding" | "submit_review",
  "path": "optional file path for read_file",
  "finding": {
    "file_path": "repo/file.py",
    "line": 1,
    "category": "bug|security|performance|maintainability|testing",
    "severity": "critical|warning|info",
    "title": "short title",
    "explanation": "why this is a real issue",
    "suggested_fix": "concrete fix"
  }
}

Rules:
- Return JSON only.
- Use read_file when you need more context.
- Use add_finding for one strong issue at a time.
- Use submit_review when you are done.
- Do not invent files or unsupported issues.
"""


def build_prompt(observation) -> str:
    """Build a deterministic prompt from the current observation."""

    opened_files = observation.metadata.get("opened_files", [])
    history_lines = [
        f"{entry.step}. {entry.operation}: {entry.summary}"
        for entry in observation.review_history[-6:]
    ]
    history_block = "\n".join(history_lines) if history_lines else "No prior actions."
    return (
        f"Task ID: {observation.task_id}\n"
        f"Difficulty: {observation.difficulty}\n"
        f"Goal: {observation.goal}\n"
        f"Repo summary: {observation.repo_summary}\n"
        f"Changed files: {', '.join(observation.changed_files)}\n"
        f"Available files: {', '.join(observation.available_files)}\n"
        f"Opened files: {', '.join(opened_files) if opened_files else 'None'}\n"
        f"Attempts remaining: {observation.attempts_remaining}\n"
        f"Current score: {observation.score:.2f}\n"
        f"Last action status: {observation.last_action_status or 'None'}\n"
        f"History:\n{history_block}\n\n"
        f"Visible context:\n{observation.visible_diff}\n"
    )


def extract_json_object(content: str) -> str:
    """Extract the first JSON object from the model response."""

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]
    return "{}"


def parse_response(content: str, observation) -> PythonReviewAction:
    """Parse the model response into a valid environment action."""

    try:
        payload = json.loads(extract_json_object(content))
    except json.JSONDecodeError:
        payload = {}

    operation = payload.get("operation")
    if operation == "read_file":
        path = payload.get("path")
        if path in observation.available_files:
            return PythonReviewAction(operation="read_file", path=path)
    elif operation == "add_finding":
        finding = payload.get("finding")
        if isinstance(finding, dict):
            try:
                return PythonReviewAction(
                    operation="add_finding",
                    finding=ReviewFinding(**finding),
                )
            except Exception:
                pass
    elif operation == "submit_review":
        return PythonReviewAction(operation="submit_review")

    return fallback_action(observation)


def fallback_action(observation) -> PythonReviewAction:
    """Choose a safe deterministic fallback action."""

    opened_files = set(observation.metadata.get("opened_files", []))
    for path in observation.available_files:
        if path not in opened_files:
            return PythonReviewAction(operation="read_file", path=path)
    return PythonReviewAction(operation="submit_review")


def request_action(client: OpenAI, observation) -> PythonReviewAction:
    """Ask the configured model for the next action."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(observation)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return parse_response(content, observation)


def make_env() -> PythonEnv:
    """Connect to a running env or start the Docker image."""

    if ENV_BASE_URL:
        return PythonEnv(base_url=ENV_BASE_URL)
    return PythonEnv.from_docker_image(DOCKER_IMAGE)


def run_task(client: OpenAI, env: PythonEnv, task_index: int) -> Dict[str, Any]:
    """Run one benchmark episode and return a reproducible result record."""

    result = env.reset()
    observation = result.observation
    step_logs: List[Dict[str, Any]] = []

    max_steps = observation.attempts_remaining
    for step in range(1, max_steps + 1):
        if result.done:
            break
        try:
            action = request_action(client, observation)
        except Exception as exc:  # noqa: BLE001
            action = fallback_action(observation)
            model_error: Optional[str] = str(exc)
        else:
            model_error = None

        result = env.step(action)
        observation = result.observation
        step_log = {
            "step": step,
            "operation": action.operation,
            "path": action.path,
            "reward": result.reward or 0.0,
            "score": observation.score,
            "done": result.done,
            "status": observation.last_action_status,
        }
        if action.finding is not None:
            step_log["finding"] = action.finding.model_dump()
        if model_error:
            step_log["model_error"] = model_error
        step_logs.append(step_log)
        print(
            f"Task {task_index} step {step}: op={action.operation} "
            f"score={observation.score:.2f} reward={(result.reward or 0.0):.2f}"
        )
        if result.done:
            break

    return {
        "task_id": observation.task_id,
        "difficulty": observation.difficulty,
        "score": observation.score,
        "steps": step_logs,
    }


def main() -> None:
    """Run the baseline across the three bundled tasks."""

    api_key = API_KEY or HF_TOKEN
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or HF_TOKEN before running inference.py")

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    env = make_env()
    results: List[Dict[str, Any]] = []

    try:
        for task_index in range(1, 4):
            results.append(run_task(client, env, task_index))
    finally:
        env.close()

    mean_score = sum(item["score"] for item in results) / len(results)
    report = {
        "model_name": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "mean_score": mean_score,
        "results": results,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()

