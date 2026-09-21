import asyncio
import random

from evals import metrics
from evals.llm_tools import OfflineLLM
from evals.sim import SimClock, simulate
from townmind.agent import Agent
from townmind.llm.base import ToolCall


def say(npc, t, text, nearby=()):
    return {"t": t, "npc": npc, "action": {"name": "say", "text": text}, "nearby": list(nearby)}


# ---------- 指标 ----------
def test_percentile():
    assert metrics.percentile([], 50) == 0.0
    assert metrics.percentile([5], 95) == 5
    assert metrics.percentile([1, 2, 3, 4, 5], 50) == 3
    assert abs(metrics.percentile([1, 2, 3, 4, 5], 95) - 4.8) < 1e-9


def test_similarity():
    assert metrics.similarity("你好呀，最近怎么样？", "你好呀，最近怎么样？") == 1.0
    assert metrics.similarity("你好", "再见了朋友") == 0.0


def test_repetition_rate_counts_own_repeats_only():
    same = [say("alice", 1, "今天法棍刚出炉，来一个吗？"), say("alice", 9, "今天法棍刚出炉，来一个吗？")]
    assert metrics.repetition_rate(same) == 0.5  # 两句里第二句重复
    other_npc = [say("alice", 1, "今天法棍刚出炉，来一个吗？"), say("bob", 9, "今天法棍刚出炉，来一个吗？")]
    assert metrics.repetition_rate(other_npc) == 0.0  # 不同 NPC 说同一句不算重复
    assert metrics.repetition_rate([]) == 0.0


def test_invented_mentions():
    assert metrics.invented_mentions("小镇的面包节快到了")  # 设定里没有面包节
    assert metrics.invented_mentions("去逛逛新开的酒馆")
    assert not metrics.invented_mentions("我去面包店买法棍")
    assert not metrics.invented_mentions("我去铁匠铺看看")
    assert not metrics.invented_mentions("周六的集市你去吗")


def test_invented_and_grounded_rates():
    says = [say("a", 1, "小镇的面包节快到了"), say("a", 2, "面粉涨价了"), say("a", 3, "你好")]
    assert abs(metrics.invented_rate(says) - 1 / 3) < 1e-9
    assert abs(metrics.grounded_rate(says) - 1 / 3) < 1e-9


def test_response_rate():
    replied = [say("alice", 10, "你好", ["bob"]), say("bob", 12, "你好呀", ["alice"])]
    assert metrics.response_rate(replied) == 0.5  # alice 的话被回应了；bob 的话没人回
    late = [say("alice", 10, "你好", ["bob"]), say("bob", 40, "你好呀", ["alice"])]
    assert metrics.response_rate(late) == 0.0  # 超过 15 秒才回，不算
    alone = [say("alice", 10, "自言自语", [])]
    assert metrics.response_rate(alone) == 0.0  # 身边没人时说的话不计入


def test_render_table_shape():
    one = metrics.aggregate([metrics.summarize([], {}, [], 60)])
    table = metrics.render_table({"a": one, "b": one})
    assert table.splitlines()[0] == "| 指标 | a | b |"
    assert len(table.splitlines()) == 2 + len(metrics.METRICS)


def run_summary(**overrides):
    base = metrics.summarize([], {}, [], 60)
    return {**base, **overrides}


def test_aggregate_mean_min_max():
    agg = metrics.aggregate([run_summary(say_count=10, repetition_rate=0.2), run_summary(say_count=20, repetition_rate=0.4)])
    assert agg["say_count"] == {"mean": 15.0, "min": 10, "max": 20, "n": 2}
    assert abs(agg["repetition_rate"]["mean"] - 0.3) < 1e-9


def test_table_cells_single_vs_multiple_runs():
    single = metrics.aggregate([run_summary(say_count=10, repetition_rate=0.25)])
    multi = metrics.aggregate([run_summary(say_count=10, repetition_rate=0.2), run_summary(say_count=20, repetition_rate=0.4)])
    t1 = metrics.render_table({"x": single})
    t2 = metrics.render_table({"x": multi})
    assert "| 说话次数 | 10 |" in t1 and "| 重复率（越低越好） | 25.0% |" in t1
    assert "| 说话次数 | 15.0 (10–20) |" in t2
    assert "| 重复率（越低越好） | 30.0% (20.0%–40.0%) |" in t2


# ---------- 仿真 ----------
def run_sim(seed=1, seconds=120, llm=None, **kw):
    random.seed(seed)
    clock = SimClock()
    agent = Agent(llm or OfflineLLM(), clock=clock, rng=random.Random(seed), **kw)
    agent.trace = []
    asyncio.run(simulate(agent, clock, seconds))
    return agent


def test_simulation_runs_and_produces_valid_actions():
    agent = run_sim()
    assert len(agent.trace) > 10
    assert agent.stats["llm_calls"] > 0
    for e in agent.trace:
        a = e["action"]
        assert a["name"] in ("move_to", "say", "idle")
        if a["name"] == "move_to":
            assert abs(a["x"]) <= 8 and abs(a["z"]) <= 8


def test_simulation_is_deterministic_for_same_seed():
    def fingerprint(agent):
        return [(e["npc"], e["action"]) for e in agent.trace]

    assert fingerprint(run_sim(seed=3)) == fingerprint(run_sim(seed=3))


def test_no_llm_baseline_makes_no_llm_calls():
    random.seed(1)
    clock = SimClock()
    agent = Agent(None, clock=clock, rng=random.Random(1))
    agent.trace = []
    asyncio.run(simulate(agent, clock, 60))
    assert agent.stats["llm_calls"] == 0 and len(agent.trace) > 0


def test_summarize_on_simulation_trace():
    agent = run_sim()
    s = metrics.summarize(agent.trace, dict(agent.stats), [0.5, 1.0], 120)
    assert s["decisions"] == len(agent.trace) and s["llm_calls"] == agent.stats["llm_calls"]
    assert s["tokens_in"] > 0 and s["latency_p50_ms"] == 750


# ---------- Agent 的评测开关与 Token 统计 ----------
def test_use_memory_false_stores_nothing():
    agent = run_sim(use_memory=False)
    assert all(store.memories == [] for store in agent._memories.values())


def test_use_lore_false_removes_lore_from_prompt():
    seen = []

    class Spy:
        async def choose_tool(self, system, user, tools):
            seen.append((system, user))
            return ToolCall("idle", {})

    run_sim(seconds=10, llm=Spy(), use_lore=False)
    assert seen and all("镇上的事" not in u and "不要编造" not in s for s, u in seen)
    seen.clear()
    run_sim(seconds=10, llm=Spy(), use_lore=True)
    assert seen and all("镇上的事" in u and "不要编造" in s for s, u in seen)


def test_token_usage_is_accumulated():
    class Fake:
        async def choose_tool(self, system, user, tools):
            return ToolCall("idle", {}, input_tokens=100, output_tokens=7)

    agent = run_sim(seconds=10, llm=Fake())
    calls = agent.stats["llm_calls"]
    assert calls > 0 and agent.stats["tokens_in"] == 100 * calls and agent.stats["tokens_out"] == 7 * calls
