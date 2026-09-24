"""evals/judge.py：不调真实模型就能测的部分——κ 的算法、标签解析、失败退化。"""
import asyncio

from evals.judge import Judge, agreement
from townmind.llm.base import ToolCall


class _FakeLLM:
    def __init__(self, calls):
        self.calls = list(calls)
        self.seen = []

    async def choose_tool(self, system, user, tools):
        self.seen.append((system, user, tools))
        c = self.calls.pop(0)
        if isinstance(c, Exception):
            raise c
        return c


def test_agreement_and_kappa_basic():
    # 4 对全一致，但两边都只打一种标签：一致率 100%，碰巧一致的期望也是 100%，κ 定义为 1
    assert agreement([("a", "a")] * 4)["kappa"] == 1.0
    # 一半一致、标签均匀：po=0.5，pe=0.5 -> κ=0
    s = agreement([("a", "a"), ("b", "b"), ("a", "b"), ("b", "a")])
    assert s["agreement"] == 0.5 and abs(s["kappa"]) < 1e-9
    # 裁判缺席的对子不算
    assert agreement([("a", None), (None, "b")])["n"] == 0


def test_kappa_penalises_chance_agreement():
    # 关键词几乎全打 unclear、裁判也几乎全打 unclear：一致率很高，κ 应该明显低于一致率
    pairs = [("unclear", "unclear")] * 9 + [("corrected", "unclear")]
    s = agreement(pairs)
    assert s["agreement"] == 0.9 and s["kappa"] < 0.5


def test_judge_label_uses_enum_tool_and_returns_reason():
    llm = _FakeLLM([ToolCall("label", {"label": "corrected", "reason": "明确否认"})])
    j = Judge(llm)
    got = asyncio.run(j.label("任务", "情境", "没有这回事", {"corrected": "否认", "unclear": "含糊"}))
    assert got == ("corrected", "明确否认")
    _, user, tools = llm.seen[0]
    assert tools[0]["parameters"]["properties"]["label"]["enum"] == ["corrected", "unclear"]
    assert "没有这回事" in user and "corrected：否认" in user


def test_judge_returns_none_on_failure_or_unknown_label():
    llm = _FakeLLM([RuntimeError("down"), ToolCall("label", {"label": "banana", "reason": ""})])
    j = Judge(llm)
    assert asyncio.run(j.label("t", "c", "r", {"a": "x"})) is None
    assert asyncio.run(j.label("t", "c", "r", {"a": "x"})) is None
    assert j.calls == 2 and j.failures == 2
