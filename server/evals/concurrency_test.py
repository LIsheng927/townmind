"""压测：同一时刻很多 NPC 都想问大模型时，asyncio.Semaphore(max_concurrent_llm) 限流
配合空间网格一起，实际跑起来是什么样子——排队等待的开销有多大、in-flight 请求数是不是
真的被卡在了并发上限以内、实际耗时跟理论下界差多远。

跑法（在 server 目录下，跟其它 evals 脚本一样用 -m 调用，这样 townmind 包才能被正确 import）：

    uv run python -m evals.concurrency_test
    uv run python -m evals.concurrency_test --counts 10 50 200 --concurrency 4 8 --latency 0.05

为了让"大家同时都想说话"这个最坏情况真实发生，所有 NPC 被撒在一小片区域内（互相都在
NEARBY_RADIUS 以内），这样每个 NPC 第一次 decide() 时 interesting=True 且 status=="ok"，
一定会去问大模型——这比真实游戏里"只有凑巧碰见的人才聊天"密集得多，是刻意构造的压力测试，
专门用来验证限流这道保险在最坏情况下也不会失效。

大模型换成一个模拟固定延迟的假实现（FakeDelayLLM，跟 tests/test_agent.py 里 FakeLLM 的
delay 参数是同一个思路），不发真实网络请求、不产生真实费用，只用来观察"排队"本身的开销。
"""
import argparse
import asyncio
import math
import random
import time

from townmind.agent import Agent
from townmind.llm.base import ToolCall

CLUSTER_RADIUS = 1.0  # 所有 NPC 撒在这个半径的方框内：对角线 < NEARBY_RADIUS(5.0)，保证互相都"在附近"


class FakeDelayLLM:
    """模拟一个有固定延迟、总是给出合法回答的大模型：只用来测排队开销，不测决策质量。"""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    async def choose_tool(self, system, user, tools):
        if self.delay:
            await asyncio.sleep(self.delay)
        return ToolCall("say", {"text": "嗯，是这样。"})


def make_cluster_positions(n: int, rng: random.Random) -> dict[str, tuple[float, float]]:
    return {
        f"npc{i}": (rng.uniform(-CLUSTER_RADIUS, CLUSTER_RADIUS), rng.uniform(-CLUSTER_RADIUS, CLUSTER_RADIUS))
        for i in range(n)
    }


async def run_once(n: int, concurrency: int, latency: float, rng: random.Random) -> dict:
    agent = Agent(FakeDelayLLM(latency), max_concurrent_llm=concurrency)
    positions = make_cluster_positions(n, rng)
    for npc_id, pos in positions.items():
        agent.update_position(npc_id, pos)

    t0 = time.perf_counter()
    await asyncio.gather(*(agent.decide(npc_id, {"pos": list(pos)}) for npc_id, pos in positions.items()))
    elapsed = time.perf_counter() - t0

    # n=1 是退化情况：只有一个 NPC 时身边不可能有别人，interesting=False，本来就不会问大模型，
    # 这不是限流失效，是"没人聊天"——expected 用它而不是死等于 n，让调用方能识别这种情况。
    expected_calls = n if n >= 2 else 0
    lower_bound = math.ceil(min(n, expected_calls) / concurrency) * latency if expected_calls else 0.0
    return {
        "n": n,
        "concurrency": concurrency,
        "elapsed": elapsed,
        "lower_bound": lower_bound,
        "llm_calls": agent.stats["llm_calls"],
        "expected_calls": expected_calls,
        "max_in_flight": agent.stats["max_in_flight"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", type=int, nargs="+", default=[4, 20, 80, 200])
    ap.add_argument("--concurrency", type=int, nargs="+", default=[4])
    ap.add_argument("--latency", type=float, default=0.05, help="每次假 LLM 调用模拟的延迟（秒）")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print(f"{'NPC数量':>8} {'并发上限':>8} {'实际耗时':>10} {'理论下界':>10} {'llm_calls':>10} {'max_in_flight':>14} {'结果':>6}")
    any_fail = False
    for concurrency in args.concurrency:
        for n in args.counts:
            r = asyncio.run(run_once(n, concurrency, args.latency, rng))
            capped = r["max_in_flight"] <= concurrency
            all_called = r["llm_calls"] == r["expected_calls"]
            ok = capped and all_called
            any_fail = any_fail or not ok
            print(
                f"{r['n']:>8} {r['concurrency']:>8} {r['elapsed']*1000:>8.1f}ms {r['lower_bound']*1000:>8.1f}ms "
                f"{r['llm_calls']:>10} {r['max_in_flight']:>14} {'OK' if ok else 'FAIL':>6}"
            )
            if not capped:
                print(f"  !! max_in_flight={r['max_in_flight']} 超过了并发上限 {concurrency}，限流没生效")
            if not all_called:
                print(f"  !! llm_calls={r['llm_calls']}，跟预期的 {r['expected_calls']} 不一致")
    if any_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
