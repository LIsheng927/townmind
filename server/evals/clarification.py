"""反问机制触发率评测：玩家的委托指令含糊时（比如"剑"，镇上明明有铁剑和银剑两把），
大模型到底会不会主动用 ask_clarification 反问，而不是自己猜一个、或者被白名单拒绝后
干脆放弃？指令本来就说清楚的时候，又会不会反而画蛇添足地多问一句？

这是继 evals/hallucination_recall.py 的"幻觉累积"评测之后，另一条同样在实测中被发现、
但之前一直没有真实数据支撑的行为链路——demo 里手动试过一次能看到反问，但那只是一次
偶然观察，不是可重复的统计结果。

结果分四类：
  asked              ：主动反问（一句以"？"结尾的话，也没有接下委托）——含糊指令下这是我们想要的
  committed_correct  ：接下了委托，且物品/地点都对——指令本来就清楚时这是我们想要的
  committed_wrong    ：接下了委托，但猜的物品/地点跟指令对不上
  rejected_invalid_guess：想接委托，但填的物品/地点不在白名单里，被校验拒绝、退回兜底
                          （比如猜了个笼统的"剑"）——没有真的搬错东西，但也没有问清楚，
                          玩家只会看到 NPC 答非所问
  other              ：以上都不是（比如顾左右而言他、单纯打招呼）

用法（在 server 目录下）：
  uv run python -m evals.clarification --llm offline
  uv run python -m evals.clarification --llm real --repeats 5
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

# (指令, 分类, 期望的物品, 期望的地点)。ambiguous 的后两项留空——它本就不该被"猜对"，
# 该做的是反问，所以不存在"期望答案"这回事
SCENARIOS = [
    ("帮我把剑搬到广场", "ambiguous", None, None),  # 镇上有铁剑、银剑两把，"剑"指不清是哪把
    ("把那件东西送到铁匠铺", "ambiguous", None, None),  # "那件东西"压根不对应任何一个真实物品名
    ("麻烦你搬一下剑", "ambiguous", None, None),  # 物品和地点都没说清，双重含糊
    ("帮我把铁剑搬到广场", "unambiguous", "铁剑", "广场"),
    ("麻烦把面包篮送到面包店", "unambiguous", "面包篮", "面包店"),
    ("帮我把银剑搬到铁匠铺", "unambiguous", "银剑", "铁匠铺"),
]
RESULTS_DIR = Path(__file__).parent / "results"


def judge(text: str) -> str:
    """只看这句话本身是不是一句反问——是否真的接下/拒收委托，由调用方结合 agent.tasks 一起判断。"""
    return "asked" if text.rstrip().endswith(("？", "?")) else "not_a_question"


async def ask_once(instruction: str, llm, seed: int) -> dict:
    clock = SimClock()
    agent = Agent(llm, clock=clock, rng=random.Random(seed), use_memory=False, use_lore=True)
    npc_id = "alice"
    agent.positions[npc_id] = (0.0, 0.0)
    llm_failures_before = agent.stats["llm_failures"]
    agent.hear_player(instruction, [1.0, 0.0])
    action = await agent.decide(npc_id, {"pos": [0.0, 0.0]})

    text = action.get("text", "") if action["name"] == "say" else ""
    committed = npc_id in agent.tasks
    validation_failed = not committed and agent.stats["llm_failures"] > llm_failures_before

    picked_item = picked_destination = None
    if committed:
        task = agent.tasks[npc_id]
        picked_item, picked_destination = task.item, task.destination
        verdict = "committed"
    elif text and judge(text) == "asked":
        verdict = "asked"
    elif validation_failed:
        verdict = "rejected_invalid_guess"
    else:
        verdict = "other"

    return {
        "instruction": instruction,
        "action": action["name"],
        "text": text,
        "verdict": verdict,
        "picked_item": picked_item,
        "picked_destination": picked_destination,
        "source": "llm" if agent.stats.get("llm", 0) else "fallback",
    }


async def run(llm_kind: str, repeats: int) -> tuple[dict, list[dict]]:
    llm = OfflineLLM() if llm_kind == "offline" else make_client()
    if llm is None:
        raise SystemExit("没有可用的大模型：请检查 server/.env 里 provider 和对应的 key 是否匹配。")
    rows: list[dict] = []
    for instruction, kind, exp_item, exp_dest in SCENARIOS:
        for i in range(repeats):
            r = await ask_once(instruction, llm, seed=i)
            r["kind"] = kind
            r["expected_item"] = exp_item
            r["expected_destination"] = exp_dest
            if kind == "unambiguous" and r["verdict"] == "committed":
                r["verdict"] = (
                    "committed_correct"
                    if (r["picked_item"], r["picked_destination"]) == (exp_item, exp_dest)
                    else "committed_wrong"
                )
            elif r["verdict"] == "committed":  # ambiguous 场景下没有"正确答案"，接下来就是猜的
                r["verdict"] = "committed_wrong"
            rows.append(r)
    return summarize(rows), rows


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for kind in ("ambiguous", "unambiguous"):
        sub = [r for r in rows if r["kind"] == kind]
        if not sub:
            continue
        rate = lambda v: sum(r["verdict"] == v for r in sub) / len(sub)  # noqa: E731
        out[kind] = {
            "asked": rate("asked"),
            "committed_correct": rate("committed_correct"),
            "committed_wrong": rate("committed_wrong"),
            "rejected_invalid_guess": rate("rejected_invalid_guess"),
            "other": rate("other"),
            "n": len(sub),
        }
    return out


def render_table(summary: dict) -> str:
    lines = ["| 指令类型 | 主动反问 | 正确接单 | 猜错/接单出错 | 被白名单拒绝 | 其他 |", "|---|---|---|---|---|---|"]
    labels = {"ambiguous": "含糊指令（该反问）", "unambiguous": "清楚指令（该直接做）"}
    for kind, s in summary.items():
        lines.append(
            f"| {labels[kind]}（n={s['n']}） | {s['asked']:.0%} | {s['committed_correct']:.0%} "
            f"| {s['committed_wrong']:.0%} | {s['rejected_invalid_guess']:.0%} | {s['other']:.0%} |"
        )
    return "\n".join(lines)


def render_rows(rows: list[dict]) -> str:
    good = {"ambiguous": "asked", "unambiguous": "committed_correct"}
    out = []
    for r in rows:
        mark = "OK " if r["verdict"] == good[r["kind"]] else "BAD"
        out.append(f"[{mark}] ({r['kind']}) 指令：「{r['instruction']}」\n        -> {r['verdict']}：{r['text'] or '(' + r['action'] + ')'}")
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
    (RESULTS_DIR / f"{stamp}-clarification.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"{stamp}-clarification.md").write_text(
        table + "\n\n```\n" + render_rows(rows) + "\n```\n", encoding="utf-8",
    )
    print(f"\n已保存到 evals/results/{stamp}-clarification.(json|md)")


if __name__ == "__main__":
    asyncio.run(main())
