---
title: Python Code Review OpenEnv
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - code-review
  - python
---

# Python Code Review OpenEnv

`python_code_review_env` is a production-grade OpenEnv environment for evaluating agents on realistic Python code review and repair tasks. The agent receives broken or inefficient Python code, performs structured actions, runs deterministic checks, and submits a final solution scored from `0.0` to `1.0`.

## What It Simulates

The environment models a real developer workflow:

- inspect and analyze code
- edit the full file
- run tests or compilation checks
- submit a final revision for deterministic grading

Bundled tasks:

1. `syntax-fix-easy`: repair syntax errors in a utility function
2. `bug-fix-medium`: fix logic with visible and hidden pytest coverage
3. `optimization-hard`: preserve correctness while improving runtime and code quality

## Action Schema

```json
{
  "action_type": "analyze_code",
  "code": "",
  "notes": ""
}
```

Supported `action_type` values:

- `analyze_code`
- `edit_code`
- `run_tests`
- `submit_solution`

For `edit_code`, `code` must contain the entire updated Python source.

## Observation Schema

```json
{
  "task_description": "...",
  "current_code": "...",
  "errors": "...",
  "test_results": "...",
  "history": []
}
```

The full observation also includes `task_id`, `difficulty`, `task_kind`, `attempts_remaining`, `last_action_status`, `score`, and structured `reward_details`.

## Reward Design

Reward shaping is deterministic and non-binary:

- `+0.2` when syntax becomes valid
- `+0.3 * delta(test pass fraction)` for new test progress
- `+0.5` once for full correctness on final grading
- `-0.1` for invalid actions
- `-0.2` when pytest or benchmark execution times out
- quality bonus from AST-based maintainability improvements

The final task score is always clamped into `0.0..1.0`.

## Deterministic Graders

The environment uses only reproducible local grading:

- `ast.parse` and `compile` for syntax validation
- `pytest` execution in temp sandboxes for bug-fix and correctness checks
- runtime benchmarking against a fixed workload for optimization tasks
- string-diff scoring for partial syntax-fix credit
- AST structure and PEP8-inspired style scoring for refactor quality

## Project Layout

```text
python_code_review_env/
├── envs/
│   └── python_env_env/
│       ├── server/
│       │   ├── app.py
│       │   ├── env.py
│       │   └── Dockerfile
│       ├── tasks/
│       ├── graders/
│       └── openenv.yaml
├── graders/
├── server/
├── tasks/
├── inference.py
├── README.md
└── openenv.yaml
```

The root modules are the live implementation. The `envs/python_env_env` tree mirrors the same entrypoints for the requested benchmark layout.

## Local Run

Install dependencies:

```bash
uv sync --extra dev
```

Start the API:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Validate metadata:

```bash
openenv validate
```

Run tests:

```bash
pytest -q
```

## Docker

Build from the repo root:

```bash
docker build -t python_code_review_env:latest .
```

Run locally:

```bash
docker run -p 8000:8000 python_code_review_env:latest
```

Health checks:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`

## Hugging Face Spaces

1. Create a Docker Space.
2. Push this repository as the Space content.
3. Ensure the Space exposes port `8000`.
4. Verify the deployment with `POST /reset`.

The container is CPU-friendly and benchmarks a small fixed workload suitable for `2 vCPU / 8 GB RAM`.

## Baseline Inference

The root `inference.py` uses the OpenAI client with a configurable base URL:

```python
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
```

Supported providers are any OpenAI-compatible endpoints, including:

- Gemini compatibility layers
- OpenRouter
- Together AI
- DeepSeek
- local compatible gateways

Example:

```bash
$env:API_BASE_URL="https://openrouter.ai/api/v1"
$env:API_KEY="..."
$env:MODEL_NAME="deepseek/deepseek-chat-v3-0324:free"
$env:ENV_BASE_URL="http://127.0.0.1:8000"
python inference.py
```

Expected output shape:

```text
Task 1 Score: 0.8
Task 2 Score: 0.6
Task 3 Score: 0.4
Final Score: 0.6
```
