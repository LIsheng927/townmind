"""评测入口。用法（在 server 目录下）：

  uv run python -m evals.run --llm offline --minutes 2          # 不花钱，只验证流程
  uv run python -m evals.run --llm real --minutes 5 --seeds 1,2,3   # 用 .env 里的真实大模型，跑 3 次取平均

会分别用几种配置跑同一个场景，输出对比表，并把结果存到 evals/results/。
"""
import argparse
import asyncio
import json
import random
from datetime import datetime
from pathlib import Path

from townmind.agent import Agent
from townmind.llm.factory import make_client

from .llm_tools import MeteredLLM, OfflineLLM
from .metrics import aggregate, annotate_says, render_flagged, render_table, summarize
from .sim import SimClock, simulate

CONFIGS = {
    "full": dict(llm=True, use_memory=True, use_lore=True),
    "no_memory": dict(llm=True, use_memory=False, use_lore=True),
    "no_lore": dict(llm=True, use_memory=True, use_lore=False),
    "no_llm": dict(llm=False, use_memory=True, use_lore=True),  # 完全不用大模型：纯规则基线
}
RESULTS_DIR = Path(__file__).parent / "results"


async def run_config(name: str, llm_kind: str, seconds: float, seed: int) -> tuple[dict, list[dict]]:
    opts = CONFIGS[name]
    random.seed(seed)  # 兜底策略用的是全局随机数，也要固定
    clock = SimClock()
    metered = None
    if opts["llm"]:
        inner = OfflineLLM() if llm_kind == "offline" else make_client()
        if inner is None:
            raise SystemExit("没有可用的大模型：请检查 server/.env 里 provider 和对应的 key 是否匹配。")
        metered = MeteredLLM(inner)
    agent = Agent(
        metered,
        clock=clock,
        rng=random.Random(seed),
        use_memory=opts["use_memory"],
        use_lore=opts["use_lore"],
    )
    agent.trace = []
    t0 = clock.t
    await simulate(agent, clock, seconds)
    summary = summarize(agent.trace, dict(agent.stats), metered.latencies if metered else [], seconds)
    return summary, annotate_says(agent.trace, t0)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["offline", "real"], default="offline")
    ap.add_argument("--minutes", type=float, default=2.0, help="每个配置模拟多少分钟的小镇时间")
    ap.add_argument("--seeds", default="1", help="用逗号分隔的多个随机种子，每个配置每个种子各跑一次")
    ap.add_argument("--configs", default="full,no_memory,no_lore,no_llm")
    args = ap.parse_args()

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"未知配置：{unknown}；可选：{list(CONFIGS)}")

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    seconds = args.minutes * 60
    raw: dict[str, list[dict]] = {}
    says_all: dict[str, dict[str, list[dict]]] = {}
    for n in names:
        raw[n], says_all[n] = [], {}
        for seed in seeds:
            print(f"[eval] 配置 {n}  seed={seed} ...", flush=True)
            summary, says = await run_config(n, args.llm, seconds, seed)
            raw[n].append(summary)
            says_all[n][str(seed)] = says
    results = {n: aggregate(runs) for n, runs in raw.items()}

    table = render_table(results)
    header = f"llm={args.llm}  每次模拟={args.minutes} 分钟  seeds={seeds}"
    if len(seeds) > 1:
        header += "\n多次运行的格式：均值 (最小–最大)。范围很宽说明这个指标噪声大，差别不能轻易当真。"
    if args.llm == "offline":
        header += "\n注意：offline 是假大模型，只验证流程，指标数字没有参考意义。"
    print("\n" + header + "\n\n" + table)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}.json").write_text(
        json.dumps(
            {"args": vars(args), "aggregate": results, "runs": raw, "says": says_all}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"{stamp}.md").write_text(header + "\n\n" + table + "\n", encoding="utf-8")
    (RESULTS_DIR / f"{stamp}-flagged.md").write_text(render_flagged(says_all), encoding="utf-8")
    print(f"\n结果已保存到 evals/results/{stamp}.(json|md)；被标记的句子见 {stamp}-flagged.md")


if __name__ == "__main__":
    asyncio.run(main())
