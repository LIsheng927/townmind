"""townmind/planner.py：游戏时钟、日程校验、模板日程、Planner 缓存。全部确定性，不调模型。"""
import pytest

from townmind import world
from townmind.planner import (
    DAY_SECONDS,
    WRITE_PLAN_TOOL,
    DailyPlan,
    GameClock,
    PlanBlock,
    Planner,
    coerce_places,
    template_plan,
)


def test_game_clock_maps_real_seconds_to_game_hours():
    c = GameClock(day_seconds=600.0, epoch=1000.0)
    assert c.day(1000.0) == 0 and c.hour(1000.0) == 0.0
    assert c.hour(1000.0 + 25.0) == pytest.approx(1.0)  # 25 秒一小时
    assert c.hour(1000.0 + 300.0) == pytest.approx(12.0)
    assert c.day(1000.0 + 600.0) == 1 and c.hour(1000.0 + 600.0) == 0.0
    assert c.describe(1000.0 + 25.0 * 14.5).startswith("下午 14:30")


def test_game_clock_reads_env(monkeypatch):
    monkeypatch.setenv("TOWNMIND_DAY_SECONDS", "1200")
    assert GameClock().day_seconds == 1200.0
    monkeypatch.delenv("TOWNMIND_DAY_SECONDS")
    assert GameClock().day_seconds == DAY_SECONDS


def test_plan_rejects_overlap_and_backwards_blocks():
    with pytest.raises(ValueError):
        DailyPlan(blocks=[PlanBlock(start=6, end=12, place="面包店", activity="a"),
                          PlanBlock(start=11, end=14, place="广场", activity="b")])
    with pytest.raises(ValueError):
        DailyPlan(blocks=[PlanBlock(start=12, end=12, place="面包店", activity="a")])
    with pytest.raises(ValueError):
        PlanBlock(start=6, end=12, place="魔法学院", activity="a")  # 地点枚举：编不出来


def test_plan_blocks_are_sorted_and_block_at_handles_gaps():
    plan = DailyPlan(blocks=[PlanBlock(start=13, end=18, place="面包店", activity="干活"),
                             PlanBlock(start=7, end=12, place="面包店", activity="烤面包")])
    assert [b.start for b in plan.blocks] == [7, 13]
    assert plan.block_at(8.5).activity == "烤面包"
    assert plan.block_at(12.5) is None  # 没排到的时段
    assert "7-12 点在面包店烤面包" in plan.describe()


def test_template_plan_is_valid_and_home_based():
    plan = template_plan("alice")
    assert plan.block_at(9).place == "面包店" and plan.block_at(3).activity == "休息"
    assert plan.block_at(19).place == "酒馆"
    # 午休/傍晚按 NPC 错开：不是十个人同一小时全去广场
    lunches = {template_plan(n).blocks[2].start for n in ("alice", "bob", "carol")}
    assert len(lunches) == 3
    # 没有工作地点的货郎：白天落到广场
    assert template_plan("milo").block_at(9).place == "广场"
    # 每个 NPC 的模板都过校验
    from townmind.personas import PERSONAS
    for npc_id in PERSONAS:
        if npc_id != "player":
            template_plan(npc_id)


def test_tool_schema_enumerates_real_places_only():
    enum = WRITE_PLAN_TOOL["parameters"]["$defs"]["PlanBlock"]["properties"]["place"]["enum"]
    assert set(enum) == {loc.name for loc in world.LOCATIONS}


def test_planner_caches_per_day_and_reports_where():
    clock = GameClock(day_seconds=600.0, epoch=0.0)
    p = Planner(clock)
    assert p.get("alice", 0.0) is None and p.where("alice", 0.0) is None
    p.set("alice", 0.0, template_plan("alice"), "template")
    assert p.where("alice", 25.0 * 9)[0] == "面包店"           # 9 点
    assert p.where("alice", 25.0 * 19)[0] == "酒馆"            # 19 点
    assert p.source("alice", 0.0) == "template"
    assert p.get("alice", 600.0) is None                        # 第二天要重新生成
    # 有日程但没排到的小时：回工作地点
    p.set("bob", 0.0, DailyPlan(blocks=[PlanBlock(start=7, end=12, place="铁匠铺", activity="打铁")]), "llm")
    assert p.where("bob", 25.0 * 15) == ("铁匠铺", "忙自己的事")


def test_coerce_places_maps_home_words_and_drops_unknown():
    raw = {"blocks": [
        {"start": 0, "end": 7, "place": "家里", "activity": "休息"},
        {"start": 7, "end": 12, "place": "铁匠铺", "activity": "打铁"},
        {"start": 12, "end": 13, "place": "去酒馆", "activity": "喝一杯"},
        {"start": 13, "end": 18, "place": "待在工作地点", "activity": "打铁"},
        {"start": 18, "end": 24, "place": "魔法学院", "activity": "上课"},
    ]}
    got = coerce_places(raw, "bob")
    assert [b["place"] for b in got["blocks"]] == ["铁匠铺", "铁匠铺", "酒馆", "铁匠铺"]
    DailyPlan(**got)  # 归一化之后能过校验
    # 没有工作地点的货郎："家"落到广场
    assert coerce_places({"blocks": [{"start": 0, "end": 7, "place": "家", "activity": "休息"}]}, "milo")["blocks"][0]["place"] == "广场"
