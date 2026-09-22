from fastapi.testclient import TestClient

from townmind import world
from townmind.agent import Agent
from townmind.main import app

app.state.agent = Agent(None)  # 测试不调用真实 LLM，走规则兜底
client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_hello_welcome():
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "hello"})
        msg = ws.receive_json()
        assert msg["type"] == "welcome"
        # 下发的地点必须和服务端的世界设定完全一致——"小镇有哪些地点"只维护一份
        assert {l["name"] for l in msg["payload"]["locations"]} == {loc.name for loc in world.LOCATIONS}


def test_observation_returns_move_action():
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "observation", "npc_id": "alice", "payload": {"pos": [0, 0]}})
        data = ws.receive_json()
        assert data["type"] == "action" and data["npc_id"] == "alice"
        assert data["payload"]["name"] == "move_to"
        assert abs(data["payload"]["x"]) <= 8 and abs(data["payload"]["z"]) <= 8


def test_bad_message_returns_error():
    with client.websocket_connect("/ws") as ws:
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"


def test_position_message_updates_position_without_reply():
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "position", "npc_id": "alice", "payload": {"pos": [3.0, 4.0]}})
        ws.send_json({"type": "hello"})  # position 没有回复，所以下一条回复应属于 hello
        assert ws.receive_json()["type"] == "welcome"
    assert app.state.agent.positions["alice"] == (3.0, 4.0)


def test_player_say_reaches_agent_without_reply():
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "player_say", "payload": {"text": "你好呀", "pos": [1.0, 1.0]}})
        ws.send_json({"type": "hello"})
        assert ws.receive_json()["type"] == "welcome"
    assert any(e.speaker == "player" and e.text == "你好呀" for e in app.state.agent.events)
