import asyncio

from townmind.agent import Agent
from townmind.breaker import CircuitBreaker
from townmind.llm.base import ToolCall


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def test_trips_after_threshold_and_recovers():
    c = Clock()
    b = CircuitBreaker(threshold=3, cooldown=30, clock=c)
    assert b.state == "closed" and b.allow()
    b.record_failure()
    b.record_failure()
    assert b.state == "closed"  # 还没到 3 次
    b.record_failure()
    assert b.state == "open" and not b.allow()
    c.t += 31
    assert b.state == "half_open"
    assert b.allow()  # 放一个试探请求
    assert not b.allow()  # 第二个不放
    b.record_success()
    assert b.state == "closed" and b.allow()


def test_failed_probe_reopens():
    c = Clock()
    b = CircuitBreaker(threshold=1, cooldown=10, clock=c)
    b.record_failure()
    c.t += 11
    assert b.allow()
    b.record_failure()
    assert b.state == "open"
    c.t += 5
    assert b.state == "open"  # 冷却重新计时


def test_success_resets_failure_count():
    b = CircuitBreaker(threshold=3, clock=Clock())
    b.record_failure()
    b.record_failure()
    b.record_success()
    b.record_failure()
    assert b.state == "closed"


class Flaky:
    def __init__(self):
        self.calls = 0
        self.down = True

    async def choose_tool(self, system, user, tools):
        self.calls += 1
        if self.down:
            raise RuntimeError("503")
        return ToolCall("say", {"text": "你好"})


def run(a):
    a.positions.setdefault("bob", (1.0, 0.0))
    return asyncio.run(a.decide("alice", {"pos": [0, 0]}))


def test_agent_stops_calling_llm_when_breaker_open_then_recovers():
    c = Clock()
    llm = Flaky()
    a = Agent(llm, clock=c, breaker=CircuitBreaker(threshold=3, cooldown=30, clock=c))
    for _ in range(3):
        run(a)
        c.t += 11  # 越过说话冷却，并让 30 秒窗口里的"说话次数"不超上限
    assert llm.calls == 3 and a.breaker.state == "open"
    run(a)
    c.t += 11
    assert llm.calls == 3 and a.stats["breaker_skipped"] == 1  # 熔断期间没有再请求
    llm.down = False
    c.t += 31
    r = run(a)  # 试探请求成功
    assert r == {"name": "say", "text": "你好"} and a.breaker.state == "closed"


def test_invalid_arguments_do_not_trip_breaker():
    class Bad:
        async def choose_tool(self, system, user, tools):
            return ToolCall("fly", {})

    c = Clock()
    a = Agent(Bad(), clock=c, breaker=CircuitBreaker(threshold=1, clock=c))
    run(a)
    assert a.breaker.state == "closed" and a.stats["llm_failures"] == 1


def test_concurrency_limit_caps_in_flight_requests():
    class Slow:
        def __init__(self):
            self.now = 0
            self.peak = 0

        async def choose_tool(self, system, user, tools):
            self.now += 1
            self.peak = max(self.peak, self.now)
            await asyncio.sleep(0.02)
            self.now -= 1
            return ToolCall("idle", {})

    llm = Slow()
    c = Clock()
    a = Agent(llm, clock=c, max_concurrent_llm=2)
    ids = [f"n{i}" for i in range(8)]
    for i, n in enumerate(ids):
        a.update_position(n, [i * 0.1, 0.0])  # 都挤在一起，彼此都在附近

    async def go():
        return await asyncio.gather(*(a.decide(n, {"pos": [0, 0]}) for n in ids))

    results = asyncio.run(go())
    assert len(results) == 8 and llm.peak == 2 and a.stats["max_in_flight"] == 2
    assert a.stats["llm_calls"] == 8  # 排队的请求最终都会被处理，不会丢
