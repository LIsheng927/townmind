import random

from townmind import fallback
from townmind.agent import Agent
from townmind.bt import Action, Condition, Selector, Sequence, Status
from townmind.llm.base import ToolCall


class Ctx:
    action = None


def yes(_):
    return True


def no(_):
    return False


def act(name):
    return Action(lambda c: {"name": name})


def test_selector_takes_first_success():
    ctx = Ctx()
    assert Selector(Sequence(Condition(no), act("a")), act("b"), act("c")).tick(ctx) is Status.SUCCESS
    assert ctx.action == {"name": "b"}


def test_sequence_stops_at_first_failure():
    ctx = Ctx()
    assert Sequence(Condition(yes), Condition(no), act("x")).tick(ctx) is Status.FAILURE
    assert ctx.action is None


def make(npc="alice", pos=(-5.0, 2.0), nearby=(), heard=False, can_say=True):
    from townmind.personas import PERSONAS

    p = PERSONAS[npc]
    return fallback.Ctx(npc, pos, p["home"], p["lines"], list(nearby), heard, can_say, random.Random(1))


def test_someone_nearby_and_can_say_speaks_persona_line():
    a = fallback.decide(make(nearby=["Bob"]))
    assert a["name"] == "say" and a["text"]
    r = fallback.decide(make(npc="bob", pos=(5, 2), nearby=["Alice"], heard=True))
    assert r["text"] in ("嗯。", "知道了，我忙。")


def test_cannot_say_does_not_speak():
    assert fallback.decide(make(nearby=["Bob"], can_say=False))["name"] != "say"


def test_away_from_home_walks_home():
    a = fallback.decide(make(pos=(0.0, 0.0)))
    assert a["name"] == "move_to" and a["place"] == "面包店"


def test_at_home_rests_or_visits_never_crashes():
    for seed in range(20):
        c = make()
        c.rng = random.Random(seed)
        assert fallback.decide(c)["name"] in ("idle", "move_to")


def test_persona_without_home_wanders():
    c = fallback.Ctx("x", (0.0, 0.0), "", {}, [], False, True, random.Random(1))
    assert fallback.decide(c)["name"] == "move_to"


def test_llm_failure_with_neighbor_falls_back_to_persona_line():
    import asyncio

    class Boom:
        async def choose_tool(self, *a):
            raise RuntimeError("down")

    a = Agent(Boom(), clock=lambda: 1000.0, rng=random.Random(1))
    a.update_position("bob", [1.0, 0.0])
    r = asyncio.run(a.decide("alice", {"pos": [0.0, 0.0]}))
    assert r["name"] == "say" and a.stats["fallback"] == 1 and a.stats["llm_failures"] == 1
