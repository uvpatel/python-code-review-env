from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


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
