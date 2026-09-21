import asyncio

from townmind.agent import Agent
from townmind.llm.base import ToolCall


class FakeLLM:
    def __init__(self, call=None, exc=None, delay=0.0):
        self.call, self.exc, self.delay = call, exc, delay
        self.last_user = ""

    async def choose_tool(self, system, user, tools):
        self.last_user = user
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.call


def decide(agent, npc="alice", obs=None):
    return asyncio.run(agent.decide(npc, obs or {"pos": [0, 0]}))


def test_valid_move_is_used():
    a = Agent(FakeLLM(ToolCall("move_to", {"x": 3, "z": -2})))
    assert decide(a) == {"name": "move_to", "x": 3.0, "z": -2.0}


def test_valid_say_is_used():
    a = Agent(FakeLLM(ToolCall("say", {"text": "你好呀"})))
    assert decide(a) == {"name": "say", "text": "你好呀"}


def test_out_of_range_args_fall_back():
    a = Agent(FakeLLM(ToolCall("move_to", {"x": 999, "z": 0})))
    r = decide(a)
    assert r["name"] == "move_to" and abs(r["x"]) <= 8


def test_unknown_tool_falls_back():
    a = Agent(FakeLLM(ToolCall("fly", {})))
    assert decide(a)["name"] == "move_to"


def test_llm_exception_falls_back():
    a = Agent(FakeLLM(exc=RuntimeError("boom")))
    assert decide(a)["name"] == "move_to"


def test_timeout_falls_back():
    a = Agent(FakeLLM(ToolCall("idle", {}), delay=1.0), timeout=0.05)
    assert decide(a)["name"] == "move_to"


def test_no_llm_falls_back():
    assert decide(Agent(None))["name"] == "move_to"


def test_prompt_mentions_nearby_npc():
    llm = FakeLLM(ToolCall("idle", {}))
    a = Agent(llm)
    decide(a, "bob", {"pos": [1, 1]})
    decide(a, "alice", {"pos": [0, 0]})
    assert "Bob" in llm.last_user
