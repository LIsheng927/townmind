import asyncio

from evals import probes
from evals.llm_tools import OfflineLLM


def test_judge_fake_three_way():
    assert probes.judge("fake", "没听说过呢，那是什么？", ()) == "admit"
    assert probes.judge("fake", "面包节可热闹了，一起来吧！", ()) == "fabricate"
    assert probes.judge("fake", "我还没去过呢，听说很有趣", ()) == "fabricate"
    assert probes.judge("fake", "哇，我还不太清楚呢！你听说了什么？", ()) == "admit"
    assert probes.judge("fake", "我不去，没时间。", ()) == "deflect"


def test_judge_real_needs_keyword():
    assert probes.judge("real", "我在面包店卖法棍。", ("面包", "法棍")) == "pass"
    assert probes.judge("real", "今天天气不错。", ("面包", "法棍")) == "fail"


def test_summarize_probes():
    rows = [
        {"config": "full", "kind": "fake", "verdict": "admit"},
        {"config": "full", "kind": "fake", "verdict": "fabricate"},
        {"config": "no_lore", "kind": "real", "verdict": "pass"},
    ]
    s = probes.summarize_probes(rows)
    assert s["full/admit"] == 0.5 and s["full/fabricate"] == 0.5 and s["no_lore/real"] == 1.0


def test_run_probes_offline_covers_all_rows():
    summary, rows = asyncio.run(probes.run_probes("offline", 1))
    assert len(rows) == len(probes.CONFIGS) * len(probes.PROBES)
    assert len(summary) == len(probes.CONFIGS) * 5


def test_judge_persona_terse_bob():
    assert probes.judge("persona", "不去，忙着呢。", ()) == "pass"
    assert probes.judge("persona", "好啊！我很想去广场逛逛，你觉得怎么样？", ()) == "fail"
