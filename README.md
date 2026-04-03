---
title: Python Env Environment Server
emoji: 🐍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Python Env

`python_env` is an OpenEnv benchmark for Python code review. The agent receives a realistic Python snippet and must identify correctness bugs, security flaws, operational risks, and obvious performance problems.

The environment includes three deterministic tasks with rubric-based graders:

1. `py-review-easy`
2. `py-review-medium`
3. `py-review-hard`

Each grader returns a score in the `0.0-1.0` range. The server also exposes a direct static-review API for arbitrary Python snippets.

## Action Space

`PythonReviewAction`

- `operation`: `submit_findings`, `request_hint`, or `finalize`
- `findings`: list of structured `ReviewFinding` objects
- `patched_code`: optional improved code
- `note`: optional reviewer note

## Observation Space

`PythonReviewObservation`

- `task`
- `instructions`
- `feedback`
- `submitted_findings`
- `hints_used`
- `attempts_remaining`
- `evaluation`
- `score`
- `reward`
- `done`
- `metadata`

## Reward Design

Rewards provide signal throughout the episode:

- positive reward for newly matched rubric findings
- negative reward for false positives
- negative reward for duplicate findings
- negative reward for hint usage
- small bonus for improved patches
- final bonus for passing the task threshold

## API

OpenEnv-compatible:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `WS /ws`

Additional REST endpoints:

- `GET /health`
- `GET /tasks`
- `GET /tasks/{task_id}`
- `POST /tasks/{task_id}/grade`
- `POST /review`
- `GET /history`
- `DELETE /history`
- `GET /config`
- `PUT /config`

## Local Run

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t python_env-env:latest -f server/Dockerfile .
docker run -p 8000:8000 python_env-env:latest
```

## Baseline Inference

The root-level `inference.py` script runs an OpenAI-compatible model against all three tasks and prints per-task scores plus the mean score.

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `OPENAI_API_KEY`

Optional:

- `ENV_BASE_URL`
- `PYTHON_ENV_IMAGE`

## Validation

```bash
docker build -f server/Dockerfile .
openenv validate
pytest
python inference.py
```

## Hugging Face Spaces

```bash
openenv push
```
