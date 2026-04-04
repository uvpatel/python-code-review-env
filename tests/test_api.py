from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert "status" in response.json()


def test_reset_returns_openenv_observation():
    response = client.post("/reset", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["observation"]["task_id"] == "py-pr-review-easy"
    assert "visible_diff" in payload["observation"]


def test_tasks_endpoint_lists_three_tasks():
    response = client.get("/tasks")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 3


def test_post_state_returns_current_state():
    client.post("/reset", json={})
    response = client.post("/state")

    assert response.status_code == 200
    payload = response.json()
    assert payload["task_id"] == "py-pr-review-easy"
    assert payload["step_count"] == 0
