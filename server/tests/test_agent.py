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
    if npc != "bob":
        agent.positions.setdefault("bob", (1.0, 1.0))  # 邻居在附近 -> 才算"有事发生"，才会问大模型
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


# ---------- NPC 之间的对话 ----------
class ScriptedLLM:
    """按顺序返回预设的工具调用，并记录每次收到的提示词。"""

    def __init__(self, calls):
        self.calls = list(calls)
        self.users = []

    async def choose_tool(self, system, user, tools):
        self.users.append(user)
        return self.calls.pop(0)


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_nearby_npc_hears_speech():
    llm = ScriptedLLM([ToolCall("say", {"text": "早上好"}), ToolCall("idle", {})])
    a = Agent(llm)
    decide(a, "alice", {"pos": [0, 0]})
    decide(a, "bob", {"pos": [1, 1]})
    assert "Alice说「早上好」" in llm.users[1]


def test_far_npc_does_not_hear_and_llm_not_called():
    llm = ScriptedLLM([ToolCall("say", {"text": "早上好"}), ToolCall("idle", {})])
    a = Agent(llm)
    decide(a, "alice", {"pos": [0, 0]})
    decide(a, "bob", {"pos": [7, 7]})
    assert len(llm.users) == 1  # bob 附近没人、也没听到话 -> 没有调用大模型
    assert a.stats["rule"] == 1


def test_speech_heard_only_once():
    llm = ScriptedLLM([ToolCall("say", {"text": "早上好"}), ToolCall("idle", {}), ToolCall("idle", {})])
    a = Agent(llm)
    decide(a, "alice", {"pos": [0, 0]})
    decide(a, "bob", {"pos": [1, 1]})
    decide(a, "bob", {"pos": [1, 1]})
    assert "你刚听到" in llm.users[1]
    assert "你刚听到" not in llm.users[2]


def test_old_speech_expires():
    clock = FakeClock()
    llm = ScriptedLLM([ToolCall("say", {"text": "早上好"}), ToolCall("idle", {})])
    a = Agent(llm, clock=clock)
    decide(a, "alice", {"pos": [0, 0]})
    clock.t += 60
    decide(a, "bob", {"pos": [1, 1]})
    assert "你刚听到" not in llm.users[1]


def test_cooldown_waits_in_place_without_calling_llm():
    clock = FakeClock()
    llm = ScriptedLLM([ToolCall("say", {"text": "一"}), ToolCall("say", {"text": "三"})])
    a = Agent(llm, clock=clock)
    assert decide(a)["name"] == "say"
    clock.t += 1  # 冷却期内
    r = decide(a)
    assert r["name"] == "idle" and 1.0 <= r["seconds"] <= 6.0  # 原地等对方回应
    assert len(llm.users) == 1  # 没有为一句注定说不了的话花钱
    clock.t += 10  # 冷却结束
    assert decide(a)["name"] == "say"
    assert len(llm.users) == 2


def test_chat_cap_makes_npc_walk_away():
    clock = FakeClock()
    llm = ScriptedLLM([ToolCall("say", {"text": str(i)}) for i in range(10)])
    a = Agent(llm, clock=clock)
    for _ in range(3):  # 30 秒窗口内说满 3 句
        assert decide(a)["name"] == "say"
        clock.t += 7
    calls_before = len(llm.users)
    r = decide(a)  # 第 4 句：已聊够，不再问大模型，走开
    assert r["name"] in ("idle", "move_to")
    assert len(llm.users) == calls_before


def test_nothing_nearby_uses_rules_not_llm():
    llm = ScriptedLLM([])
    a = Agent(llm)
    r = asyncio.run(a.decide("alice", {"pos": [0, 0]}))  # 世界里只有 alice 自己
    assert r["name"] in ("idle", "move_to")
    assert llm.users == [] and a.stats["llm_calls"] == 0


def test_stats_count_llm_calls_and_failures():
    a = Agent(FakeLLM(exc=RuntimeError("boom")))
    decide(a)
    assert a.stats["llm_calls"] == 1 and a.stats["llm_failures"] == 1 and a.stats["fallback"] == 1


def test_end_conversation_becomes_farewell_then_walks_away():
    clock = FakeClock()
    llm = ScriptedLLM([ToolCall("end_conversation", {"farewell": "再见！"}), ToolCall("say", {"text": "不该被问到"})])
    a = Agent(llm, clock=clock)
    assert decide(a) == {"name": "say", "text": "再见！"}  # 对 Unity 来说就是说了一句话
    assert a.stats["ended_conversations"] == 1
    clock.t += 7  # 冷却已过，但还在"离开"状态
    r = decide(a)
    assert r["name"] in ("idle", "move_to")
    assert len(llm.users) == 1  # 没再问大模型
    clock.t += 30  # 离开状态结束
    a.llm = ScriptedLLM([ToolCall("idle", {})])
    decide(a)
    assert len(a.llm.users) == 1  # 又恢复了正常决策


def test_farewell_is_heard_by_neighbor():
    llm = ScriptedLLM([ToolCall("end_conversation", {"farewell": "我先走了"}), ToolCall("idle", {})])
    a = Agent(llm)
    decide(a, "alice", {"pos": [0, 0]})
    decide(a, "bob", {"pos": [1, 1]})
    assert "我先走了" in llm.users[1]


def test_prompt_tells_how_many_lines_already_said():
    clock = FakeClock()
    llm = ScriptedLLM([ToolCall("say", {"text": "你好"}), ToolCall("idle", {})])
    a = Agent(llm, clock=clock)
    decide(a)
    clock.t += 7
    decide(a)
    assert "已经说了 1 句话" in llm.users[1]


def test_empty_farewell_falls_back():
    a = Agent(FakeLLM(ToolCall("end_conversation", {"farewell": ""})))
    assert decide(a)["name"] == "move_to"


# ---------- 记忆 ----------
def texts(agent, npc="alice"):
    return [m.text for m in agent._mem(npc).memories]


def test_first_meeting_is_remembered_only_once():
    a = Agent(FakeLLM(ToolCall("idle", {})))
    decide(a)
    decide(a)
    assert sum("第一次见到Bob" in t for t in texts(a)) == 1
    assert "bob" in a._mem("alice").met


def test_recalled_memory_appears_in_next_prompt():
    clock = FakeClock()
    llm = ScriptedLLM([ToolCall("say", {"text": "你好"}), ToolCall("idle", {})])
    a = Agent(llm, clock=clock)
    decide(a)
    clock.t += 7
    decide(a)
    assert "你想起了" in llm.users[1]
    assert "你对Bob说了「你好」" in llm.users[1]


def test_heard_speech_becomes_important_memory_about_speaker():
    llm = ScriptedLLM([ToolCall("say", {"text": "早上好"}), ToolCall("idle", {})])
    a = Agent(llm)
    decide(a, "alice", {"pos": [0, 0]})
    decide(a, "bob", {"pos": [1, 1]})
    heard = [m for m in a._mem("bob").memories if "对你说" in m.text]
    assert len(heard) == 1 and heard[0].people == frozenset({"alice"}) and heard[0].importance == 6


def test_memory_survives_restart(tmp_path):
    a1 = Agent(FakeLLM(ToolCall("idle", {})), memory_dir=tmp_path)
    decide(a1)
    a2 = Agent(None, memory_dir=tmp_path)  # 模拟服务重启
    assert any("第一次见到Bob" in t for t in texts(a2))
    assert "bob" in a2._mem("alice").met


def test_idle_is_not_worth_remembering():
    a = Agent(FakeLLM(ToolCall("idle", {})))
    decide(a)
    assert not any("休息" in t for t in texts(a))
