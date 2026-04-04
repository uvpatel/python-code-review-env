---
title: Python PR Review OpenEnv
emoji: 🐍
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - code-review
---

# Python PR Review OpenEnv

`python_env` is a deterministic OpenEnv benchmark for real-world Python pull request review. The agent receives a realistic diff, can request full file context, records structured review findings, and submits a final review. The environment scores the review locally with deterministic rubric matchers instead of LLM graders.

## Why this is useful

The benchmark models a real workflow humans do every day:

- inspect a partial diff
- read additional files when the diff is not enough
- decide which findings are real defects versus noise
- submit a review with precise file and line references

This makes it suitable for evaluating agentic code-review behavior, trajectory reward shaping, and precision under partial context.

## Tasks

The environment ships with exactly three bundled tasks:

1. `py-pr-review-easy`: single-file regression in a retry helper
2. `py-pr-review-medium`: multi-file billing change with a correctness bug and missing test coverage
3. `py-pr-review-hard`: async job-runner change with subtle concurrency problems

Each task has a deterministic hidden rubric with weights that sum to `1.0`. Scores are computed as:

`matched_weight - false_positive_penalties - duplicate_penalties`

The final score is clamped to `0.0-1.0`.

## Action Space

`PythonReviewAction`

- `operation`: `read_file | add_finding | submit_review | finish`
- `path`: repository-relative path for `read_file`
- `finding`: structured `ReviewFinding` for `add_finding`

`ReviewFinding`

- `file_path`
- `line`
- `category`: `bug | security | performance | maintainability | testing`
- `severity`: `critical | warning | info`
- `title`
- `explanation`
- `suggested_fix`

## Observation Space

`PythonReviewObservation`

- `task_id`
- `difficulty`
- `goal`
- `repo_summary`
- `changed_files`
- `visible_diff`
- `available_files`
- `review_history`
- `attempts_remaining`
- `last_action_status`
- `score`
- `reward_details`
- OpenEnv base fields: `done`, `reward`, `metadata`

## Reward Design

The reward is shaped across the full trajectory:

- positive reward for opening new relevant files
- positive reward equal to the matched rubric weight for a newly accepted finding
- negative reward for false positives
- negative reward for duplicate findings
- negative reward for repeated file reads and step inefficiency
- terminal reward that includes the final normalized task score

The scalar OpenEnv reward is exposed in `observation.reward`, while the structured breakdown is exposed in `observation.reward_details`.

## API

OpenEnv-compatible routes:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /health`

Additional helper routes:

- `POST /state`
- `GET /tasks`
- `GET /tasks/{task_id}`
- `POST /tasks/{task_id}/grade`
- `GET /healthz`

## Local Development

Install dependencies:

```bash
uv sync
```

Run the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run tests:

```bash
pytest
```

Validate OpenEnv metadata:

```bash
openenv validate
```

## Baseline Inference

The required root script is `inference.py`. It uses the OpenAI client only and reads:

- `OPENAI_API_KEY`
- `API_BASE_URL`
- `MODEL_NAME`
- optional `HF_TOKEN`
- optional `ENV_BASE_URL`

Example:

```bash
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4.1-mini"
$env:OPENAI_API_KEY="..."
$env:ENV_BASE_URL="http://127.0.0.1:8000"
python inference.py
```

The script runs all three tasks in fixed order, prints per-task progress, and writes `inference_results.json`.

## Docker

Build:

```bash
docker build -t python_env-env:latest .
```

Run:

```bash
docker run -p 8000:8000 python_env-env:latest
```

## Hugging Face Spaces

This repository is configured for Docker Spaces. The root `Dockerfile` starts `uvicorn server.app:app` on `$PORT`, and `POST /reset` returns a valid observation for validator pings.
