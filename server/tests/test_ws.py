from fastapi.testclient import TestClient

from townmind.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_hello_welcome():
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello"})
        assert ws.receive_json()["type"] == "welcome"


def test_observation_returns_idle_action():
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "observation", "npc_id": "alice", "payload": {"pos": [0, 0]}})
        data = ws.receive_json()
        assert data["type"] == "action" and data["npc_id"] == "alice"


def test_bad_message_returns_error():
    with client.websocket_connect("/ws") as ws:
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
