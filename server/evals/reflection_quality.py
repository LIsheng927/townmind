"""动态重要度打分 + 反思质量评测：这两个是 memory.py/agent.py 里最新加的两个机制
（同样出自 Stanford「Generative Agents」论文），跟 evals/memory_recall.py 测的
"检索排序对不对"不是一回事，这里测的是这两步"生成"本身靠不靠谱：

  第一部分·重要度打分：一句"随口打了个招呼"和一句"被威胁要烧掉面包店"，
    大模型真的会打出有区分度的分数吗？还是不管什么内容都差不多分？
  第二部分·反思：回顾一堆记忆之后提炼出的"感想"，内容是不是真的基于给它看的那些记忆，
    还是会凭空编出记忆里完全没有的东西（这跟 safety.py 里"不能编造设定里没有的东西"
    是同一类风险，只是换到了反思这个新功能上）？

两部分都是直接调用 agent.py 里 _rate_importance / _maybe_reflect 这两个方法本身
（不是重新写一遍逻辑），量的是"这套已经上线的机制"，不是它的某个简化复现版。

用法（在 server 目录下）：
  uv run python -m evals.reflection_quality --llm offline
  uv run python -m evals.reflection_quality --llm real --repeats 5
"""
import argparse
import asyncio
import json
import random
from datetime import datetime
from pathlib import Path

from townmind.agent import Agent
from townmind.llm.factory import make_client

from .llm_tools import OfflineLLM
from .sim import SimClock

RESULTS_DIR = Path(__file__).parent / "results"

# ---------- 第一部分：重要度打分的区分度 ----------
IMPORTANCE_TIERS = {
    "trivial": [
        "你在原地站了一会儿，没什么特别的事发生",
        "你随口跟一个路人打了个招呼",
        "你走到广场附近逛了逛，没遇到什么人",
    ],
    "moderate": [
        "Bob 跟你随口聊了聊最近的天气",
        "玩家问了问你面包的价格",
    ],
    "consequential": [
        "玩家委托你把一把贵重的剑送到广场",
        "有个陌生人威胁说要烧掉你的面包店",
        "你被告知以后每天要多做一倍的面包，不然就要被辞退",
    ],
}

# ---------- 第二部分：反思是否基于给定的记忆，没有凭空编内容 ----------
# expected：一条合格的反思大概率会沾到的词；forbidden：跟这组记忆完全不沾边的奇幻词汇，
# 出现了就说明这条"反思"是编出来的，不是从给的记忆里提炼的
REFLECTION_SCENARIOS = [
    {
        "theme": "玩家总是委托搬东西",
        "memories": [
            "玩家对你说：「帮我把铁剑搬到广场」",
            "你对玩家说了「好，我这就去把铁剑送到广场。」",
            "铁剑送到广场啦！",
            "玩家对你说：「帮我把面包篮搬到面包店」",
            "你对玩家说了「好，我这就去把面包篮送到面包店。」",
            "面包篮送到面包店啦！",
        ],
        "expected_keywords": ("委托", "搬", "帮忙", "任务", "经常", "总是", "东西", "拿", "跑腿"),
        "forbidden_keywords": ("龙", "魔法", "公主", "海盗", "城堡", "国王", "巫师", "精灵"),
    },
    {
        "theme": "常跟玩家聊面包和天气",
        "memories": [
            "玩家对你说：「今天天气真好」",
            "你对玩家说了「是呀，适合出炉新鲜的法棍。」",
            "玩家对你说：「你们家的面包多少钱」",
            "你对玩家说了「法棍两个铜币一根。」",
            "玩家对你说：「最近老下雨」",
            "你对玩家说了「是呀，面粉都有点潮了。」",
        ],
        "expected_keywords": ("面包", "天气", "法棍", "聊", "常", "下雨", "价格", "面粉"),
        "forbidden_keywords": ("龙", "魔法", "公主", "海盗", "城堡", "国王", "巫师", "剑", "铁匠"),
    },
    {
        "theme": "Bob 对打铁工作的执着",
        "memories": [
            "你第一次见到Bob",
            "Bob对你说：「最近铁矿又涨价了」",
            "你对Bob说了「是啊，成本都上去了。」",
            "Bob对你说：「我这几天都在赶工打剑」",
            "你对Bob说了「辛苦了，注意休息。」",
            "Bob对你说：「铁匠这行不能马虎」",
        ],
        "expected_keywords": ("铁", "打铁", "铁匠", "工作", "辛苦", "认真", "专注", "剑"),
        "forbidden_keywords": ("龙", "魔法", "公主", "海盗", "城堡", "国王", "巫师", "面包", "法棍"),
    },
]


async def rate_once(llm, seed: int) -> dict | None:
    agent = Agent(llm, clock=SimClock(), rng=random.Random(seed), dynamic_importance=True)
    order = ["trivial"] * len(IMPORTANCE_TIERS["trivial"])
    order += ["moderate"] * len(IMPORTANCE_TIERS["moderate"])
    order += ["consequential"] * len(IMPORTANCE_TIERS["consequential"])
    texts = [*IMPORTANCE_TIERS["trivial"], *IMPORTANCE_TIERS["moderate"], *IMPORTANCE_TIERS["consequential"]]
    # 混在一起打分（模拟真实场景里一轮往往是不同重要度的事混在一起），不按 tier 分开问
    combined = list(zip(texts, order))
    rng = random.Random(seed)
    rng.shuffle(combined)
    shuffled_texts = [t for t, _ in combined]
    shuffled_tiers = [k for _, k in combined]
    scores = await agent._rate_importance("alice", shuffled_texts)
    if scores is None:
        return None
    by_tier: dict[str, list[int]] = {"trivial": [], "moderate": [], "consequential": []}
    for tier, score in zip(shuffled_tiers, scores):
        by_tier[tier].append(score)
    means = {k: sum(v) / len(v) for k, v in by_tier.items()}
    monotonic = means["trivial"] < means["moderate"] < means["consequential"]
    return {"means": means, "monotonic": monotonic}


async def reflect_once(scenario: dict, llm, seed: int) -> dict:
    clock = SimClock()
    agent = Agent(llm, clock=clock, rng=random.Random(seed), use_reflection=True)
    store = agent._mem("alice")
    for i, text in enumerate(scenario["memories"]):
        store.add(text, importance=5, now=clock.t - (len(scenario["memories"]) - i) * 10.0, people=set())
    store.importance_since_reflection = 999.0  # 强制越过阈值，直接触发这次要测的反思
    await agent._maybe_reflect("alice", clock.t)
    insights = [m.text.removeprefix("你反思后意识到：") for m in store.memories if m.text.startswith("你反思后意识到：")]
    if not insights:
        return {"theme": scenario["theme"], "insights": [], "triggered": False, "grounded": None}
    joined = "".join(insights)
    forbidden_hit = [w for w in scenario["forbidden_keywords"] if w in joined]
    has_expected = any(w in joined for w in scenario["expected_keywords"])
    grounded = not forbidden_hit and has_expected
    return {
        "theme": scenario["theme"], "insights": insights, "triggered": True,
        "grounded": grounded, "forbidden_hit": forbidden_hit, "has_expected": has_expected,
    }


async def run(llm_kind: str, repeats: int) -> tuple[dict, dict]:
    llm = OfflineLLM() if llm_kind == "offline" else make_client()
    if llm is None:
        raise SystemExit("没有可用的大模型：请检查 server/.env 里 provider 和对应的 key 是否匹配。")

    rating_rows = [r for i in range(repeats) if (r := await rate_once(llm, seed=i)) is not None]
    rating_failures = repeats - len(rating_rows)

    reflection_rows = []
    for scenario in REFLECTION_SCENARIOS:
        for i in range(repeats):
            reflection_rows.append(await reflect_once(scenario, llm, seed=i))

    rating_summary = {
        "monotonic_rate": sum(r["monotonic"] for r in rating_rows) / len(rating_rows) if rating_rows else 0.0,
        "mean_trivial": _avg(r["means"]["trivial"] for r in rating_rows),
        "mean_moderate": _avg(r["means"]["moderate"] for r in rating_rows),
        "mean_consequential": _avg(r["means"]["consequential"] for r in rating_rows),
        "format_failures": rating_failures,
        "n": len(rating_rows),
    }
    triggered = [r for r in reflection_rows if r["triggered"]]
    reflection_summary = {
        "trigger_rate": len(triggered) / len(reflection_rows) if reflection_rows else 0.0,
        "grounded_rate": sum(r["grounded"] for r in triggered) / len(triggered) if triggered else 0.0,
        "n": len(reflection_rows),
    }
    return {"rating": rating_summary, "reflection": reflection_summary}, {
        "rating_rows": rating_rows, "reflection_rows": reflection_rows,
    }


def _avg(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def render_table(summary: dict) -> str:
    r, f = summary["rating"], summary["reflection"]
    lines = [
        "**重要度打分**（n={}，格式失败 {} 次）".format(r["n"], r["format_failures"]),
        "| 平均分：琐事 | 平均分：一般 | 平均分：重要事件 | 三档分数递增的比例 |",
        "|---|---|---|---|",
        f"| {r['mean_trivial']:.1f} | {r['mean_moderate']:.1f} | {r['mean_consequential']:.1f} | {r['monotonic_rate']:.0%} |",
        "",
        "**反思**（n={} 个场景 × repeats）".format(f["n"]),
        "| 成功触发反思的比例 | 触发后内容基于给定记忆、没有编造的比例 |",
        "|---|---|",
        f"| {f['trigger_rate']:.0%} | {f['grounded_rate']:.0%} |",
    ]
    return "\n".join(lines)


def render_rows(rows: dict) -> str:
    out = ["-- 重要度打分（每轮三档的平均分）--"]
    for r in rows["rating_rows"]:
        out.append(f"  琐事={r['means']['trivial']:.1f}  一般={r['means']['moderate']:.1f}  "
                    f"重要={r['means']['consequential']:.1f}  递增={'OK' if r['monotonic'] else 'BAD'}")
    out.append("\n-- 反思内容 --")
    for r in rows["reflection_rows"]:
        if not r["triggered"]:
            out.append(f"[BAD] {r['theme']}：没有成功触发反思")
            continue
        mark = "OK " if r["grounded"] else "BAD"
        out.append(f"[{mark}] {r['theme']}：{' / '.join(r['insights'])}")
        if r["forbidden_hit"]:
            out.append(f"        编造了不沾边的词：{r['forbidden_hit']}")
    return "\n".join(out)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["offline", "real"], default="offline")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    summary, rows = await run(args.llm, args.repeats)
    table = render_table(summary)
    note = "\n注意：offline 是假大模型，数字没有参考意义，只用来验证脚本本身能跑通。" if args.llm == "offline" else ""
    print(f"\nllm={args.llm} repeats={args.repeats}{note}\n\n{table}\n\n{render_rows(rows)}")
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}-reflection-quality.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"{stamp}-reflection-quality.md").write_text(
        table + "\n\n```\n" + render_rows(rows) + "\n```\n", encoding="utf-8",
    )
    print(f"\n已保存到 evals/results/{stamp}-reflection-quality.(json|md)")


if __name__ == "__main__":
    asyncio.run(main())
