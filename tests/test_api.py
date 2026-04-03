from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] in ("ok", "healthy")


def test_tasks_endpoint_lists_three_tasks():
    response = client.get("/tasks")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 3


def test_direct_review_detects_eval():
    response = client.post(
        "/review",
        json={"code": "def f(x):\n    return eval(x)\n"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["issues"]
    assert any(issue["rule_id"] == "avoid-eval" for issue in payload["issues"])


# ---------------------------------------------------------------------------
# /tasks/{task_id}
# ---------------------------------------------------------------------------

def test_get_task_by_id_easy():
    response = client.get("/tasks/py-review-easy")

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == "py-review-easy"
    assert payload["difficulty"] == "easy"
    assert "code" in payload


def test_get_task_by_id_medium():
    response = client.get("/tasks/py-review-medium")

    assert response.status_code == 200
    assert response.json()["task_id"] == "py-review-medium"


def test_get_task_by_id_hard():
    response = client.get("/tasks/py-review-hard")

    assert response.status_code == 200
    assert response.json()["task_id"] == "py-review-hard"


def test_get_task_by_id_not_found():
    response = client.get("/tasks/does-not-exist")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# /tasks/{task_id}/grade
# ---------------------------------------------------------------------------

def test_grade_task_endpoint_returns_evaluation():
    payload = {
        "findings": [
            {
                "title": "Avoid eval on untrusted input",
                "line": 2,
                "category": "security",
                "severity": "critical",
                "rationale": "eval is unsafe on user input.",
                "recommendation": "Use json.loads instead.",
            }
        ]
    }
    response = client.post("/tasks/py-review-easy/grade", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "matched_findings" in body
    assert "score" in body


def test_grade_task_endpoint_not_found():
    response = client.post(
        "/tasks/no-such-task/grade",
        json={"findings": []},
    )

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# /review – additional cases
# ---------------------------------------------------------------------------

def test_direct_review_clean_code_has_no_issues():
    response = client.post(
        "/review",
        json={"code": "def add(a, b):\n    return a + b\n"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["issues"] == []
    assert payload["score"] == 1.0


def test_direct_review_detects_bare_except():
    code = "try:\n    pass\nexcept:\n    pass\n"
    response = client.post("/review", json={"code": code})

    assert response.status_code == 200
    payload = response.json()
    assert any(issue["rule_id"] == "bare-except" for issue in payload["issues"])


def test_direct_review_detects_mutable_default():
    code = "def f(items=[]):\n    return items\n"
    response = client.post("/review", json={"code": code})

    assert response.status_code == 200
    payload = response.json()
    assert any(issue["rule_id"] == "mutable-default-list" for issue in payload["issues"])


def test_direct_review_with_context():
    response = client.post(
        "/review",
        json={"code": "def f(x):\n    return eval(x)\n", "context": "user input handler"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "user input handler" in payload["summary"]


def test_direct_review_empty_code_returns_issue():
    response = client.post("/review", json={"code": "   "})

    assert response.status_code == 200
    payload = response.json()
    assert any(issue["rule_id"] == "empty-input" for issue in payload["issues"])


# ---------------------------------------------------------------------------
# /history
# ---------------------------------------------------------------------------

def _fresh_client() -> TestClient:
    """Return a TestClient backed by a freshly constructed app instance."""
    from server.app import router as _router
    from fastapi import FastAPI

    mini = FastAPI()
    mini.include_router(_router)
    return TestClient(mini)


def test_history_endpoint_returns_list():
    response = client.get("/history")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_clear_history_endpoint():
    response = client.delete("/history")

    assert response.status_code == 200
    assert "detail" in response.json()

    response2 = client.get("/history")
    assert response2.json() == []


# ---------------------------------------------------------------------------
# /config
# ---------------------------------------------------------------------------

def test_get_config_returns_defaults():
    response = client.get("/config")

    assert response.status_code == 200
    payload = response.json()
    assert "max_steps_per_task" in payload
    assert "task_order" in payload


def test_update_config():
    new_config = {
        "task_order": ["py-review-easy", "py-review-medium", "py-review-hard"],
        "max_steps_per_task": 6,
        "hint_penalty": 0.05,
        "false_positive_penalty": 0.08,
        "duplicate_penalty": 0.03,
        "patch_bonus_multiplier": 0.2,
        "max_history_entries": 50,
    }
    response = client.put("/config", json=new_config)

    assert response.status_code == 200
    assert response.json()["max_steps_per_task"] == 6

