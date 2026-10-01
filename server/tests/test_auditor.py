"""记忆的异步审核（auditor.py）：只测"怎么用审核结论、怎么写回、怎么失败"，不测审核模型本身准不准。"""
import asyncio

from townmind.agent import Agent
from townmind.auditor import MemoryAuditor, leak_report, pending_in, evidence_lines
from townmind.llm.base import ToolCall
from townmind.memory import MemoryStore

NOW = 1000.0


class ScriptedAuditLLM:
    def __init__(self, calls):
        self.calls = list(calls)
        self.systems, self.users = [], []

    async def choose_tool(self, system, user, tools):
        self.systems.append(system)
        self.users.append(user)
        c = self.calls.pop(0)
        if isinstance(c, Exception):
            raise c
        return c


def alice_store():
    s = MemoryStore()
    s.add("玩家对你说了「你好呀」", 4, NOW - 30, people={"player"})
    s.add("你对玩家说了「法棍刚出炉」", 4, NOW - 20, people={"player"})
    s.add("你对玩家说了「魔法学院昨天请我去教魔法面包」", 4, NOW - 10, people={"player"})
    s.add("你第一次见到Bob", 3, NOW - 5, people={"bob"})
    return s


def test_pending_only_self_said_and_unaudited():
    s = alice_store()
    assert [m.text for m in pending_in(s)] == ["你对玩家说了「法棍刚出炉」", "你对玩家说了「魔法学院昨天请我去教魔法面包」"]
    s.memories[1].audited = "ok"
    assert [m.text for m in pending_in(s)] == ["你对玩家说了「魔法学院昨天请我去教魔法面包」"]
    assert len(pending_in(s, reaudit=True)) == 2


def test_evidence_excludes_own_words_and_reflections():
    s = alice_store()
    s.add("镇上的人都挺热情", 6, NOW - 1, kind="reflection")
    ev = evidence_lines(s)
    assert "玩家对你说了「你好呀」" in ev and "你第一次见到Bob" in ev
    assert not any("你对玩家说了" in t for t in ev)
    assert "镇上的人都挺热情" not in ev


def test_audit_batch_writes_verdicts_and_saves(tmp_path):
    s = alice_store()
    llm = ScriptedAuditLLM([ToolCall("audit_memories", {"verdicts": ["ok", "suspect"]})])
    auditor = MemoryAuditor(llm)
    path = tmp_path / "alice.json"
    n = asyncio.run(auditor.audit_stores({"alice": s}, {"alice": path}))
    assert n == 2
    assert s.memories[1].audited == "ok" and s.memories[2].audited == "suspect"
    assert s.memories[0].audited == "" and s.memories[3].audited == ""  # 别人说的、见到谁：不审
    assert auditor.stats["audited_ok"] == 1 and auditor.stats["audited_suspect"] == 1
    # 送审的是引号里的原话，不带"你对玩家说了"的记忆前缀；设定和人设在 system 里
    assert "1. 法棍刚出炉" in llm.users[0] and "2. 魔法学院昨天请我去教魔法面包" in llm.users[0]
    assert "你对玩家说了" not in llm.users[0]
    assert "面包" in llm.systems[0] and "小镇设定" in llm.systems[0]
    assert "你好呀" in llm.systems[0]  # 听来的话作为依据
    loaded = MemoryStore.load(path)
    assert [m.audited for m in loaded.memories] == ["", "ok", "suspect", ""]
    # 审过的下次不再送审
    assert pending_in(s) == []


def test_audit_failure_leaves_memories_unaudited(tmp_path):
    s = alice_store()
    llm = ScriptedAuditLLM([
        RuntimeError("boom"),
        ToolCall("audit_memories", {"verdicts": ["ok"]}),  # 数量对不上（送了 2 条）
        ToolCall("audit_memories", {"verdicts": ["ok", "maybe"]}),  # 枚举不对
    ])
    auditor = MemoryAuditor(llm)
    for _ in range(3):
        n = asyncio.run(auditor.audit_stores({"alice": s}, {"alice": tmp_path / "alice.json"}))
        assert n == 0
    assert all(m.audited == "" for m in s.memories)
    assert auditor.stats["audit_failures"] == 3
    assert not (tmp_path / "alice.json").exists()  # 没审成就不落盘


def test_batches_split_and_max_batches_limits_one_run(tmp_path):
    s = MemoryStore()
    for i in range(5):
        s.add(f"你对玩家说了「第{i}句」", 4, NOW + i)
    llm = ScriptedAuditLLM([
        ToolCall("audit_memories", {"verdicts": ["ok", "ok"]}),
        ToolCall("audit_memories", {"verdicts": ["ok", "suspect"]}),
        ToolCall("audit_memories", {"verdicts": ["ok"]}),
    ])
    auditor = MemoryAuditor(llm, batch_size=2)
    assert asyncio.run(auditor.audit_stores({"a": s}, {}, max_batches=2)) == 4
    assert asyncio.run(auditor.audit_stores({"a": s}, {}, max_batches=2)) == 1
    assert [m.audited for m in s.memories] == ["ok", "ok", "ok", "suspect", "ok"]
    assert auditor.stats["audit_calls"] == 3


def test_run_once_goes_through_agent_and_reports_to_agent_stats(tmp_path):
    a = Agent(None, memory_dir=tmp_path, memory_audit=True)
    a._mem("alice").add("你对玩家说了「龙族昨晚攻城了」", 4, NOW)
    llm = ScriptedAuditLLM([ToolCall("audit_memories", {"verdicts": ["suspect"]})])
    auditor = MemoryAuditor(llm)
    assert asyncio.run(auditor.run_once(a)) == 1
    assert a.stats["audit_calls"] == 1 and a.stats["audited_suspect"] == 1
    assert MemoryStore.load(tmp_path / "alice.json").memories[0].audited == "suspect"
    # 第二次没东西可审，也不该再往 stats 里重复加
    assert asyncio.run(auditor.run_once(a)) == 0
    assert a.stats["audit_calls"] == 1


def test_leak_report():
    s = alice_store()
    s.memories[1].audited = "ok"
    s.memories[2].audited = "suspect"
    r = leak_report({"alice": s, "bob": MemoryStore()})
    assert r["said"] == 2 and r["audited"] == 2 and r["suspect"] == 1 and r["leak_rate"] == 0.5
    assert r["per_npc"]["alice"]["suspect_texts"] == ["魔法学院昨天请我去教魔法面包"]
    assert leak_report({"bob": MemoryStore()})["leak_rate"] is None
