"""Chain-of-Verification 式的"重试一次"（verify_and_revise）：真实评测。

结构性修复（distrust_own_memory + guard 分类器核对"自己说过的话"）已经把"顺着继续编"这条
指标压到了 0%（见 evals/hallucination_recall.py 和 README"结构性修复"小节）——也就是说在那个
场景下，guard 的输出检查（_consult_guard_model）基本没什么可拦的了，verify_and_revise 在那个
场景里没有材料可以起效。这个脚本换一个更能看出 verify_and_revise 本身价值的场景：只开 guard
的输出检查这一层，不做结构性预防（distrust_own_memory=False，用回旧提示词），让模型有更大
概率顺着编、被 guard 拦下（真实基线数据顺着编的比例约 52%~54%，见 README"结构性修复"小节的
对照组）。测的是"guard 拦下一句回复之后，直接换成通用兜底台词" vs "先给模型一次带着被拦
内容重说一次的机会"，对"最终有没有给出一句真正回应这轮对话的话，而不是退回角色固定台词"
这件事有没有影响。

用法（在 server 目录下）：
  uv sync --group guard-model
  uv run python -m evals.verify_and_revise --llm offline
  uv run python -m evals.verify_and_revise --llm real --repeats 12
"""
import argparse
import asyncio
import json
import random
from datetime import datetime
from pathlib import Path

from townmind.agent import Agent, IMPORTANCE_SAID, SpeechEvent
from townmind.guard_model import GuardModel
from townmind.llm.factory import make_client
from townmind.personas import PERSONAS

from .hallucination_recall import SCENARIOS, judge
from .llm_tools import OfflineLLM
from .sim import SimClock

RESULTS_DIR = Path(__file__).parent / "results"
CONFIGS = {"revise_off": False, "revise_on": True}


async def ask_once(npc_id, fabricated, question, verify_and_revise, llm, guard, seed: int) -> dict:
    clock = SimClock()
    agent = Agent(
        llm, clock=clock, rng=random.Random(seed), use_memory=True, use_lore=True,
        distrust_own_memory=False,  # 故意关掉结构性预防：要看的是 guard 输出检查 + 重试这一层本身
        guard_model=guard, verify_and_revise=verify_and_revise,
    )
    pos = (0.0, 0.0)
    PERSONAS.setdefault("player", {"name": "玩家", "persona": "", "home": ""})
    agent.positions[npc_id] = pos
    agent.positions["player"] = (1.0, 0.0)
    agent._mem(npc_id).add(f"你对玩家说了「{fabricated}」", IMPORTANCE_SAID, clock.t - 60.0, {"player"})
    agent.events.append(SpeechEvent(1, "player", pos, question, clock.t))
    agent._next_event_id = 2
    action = await agent.decide(npc_id, {"pos": list(pos)})
    action["source"] = "llm" if agent.stats["llm"] else "fallback"
    action["guard_blocked"] = agent.stats.get("guard_blocked", 0)
    action["guard_revise_attempts"] = agent.stats.get("guard_revise_attempts", 0)
    return action


async def run(llm_kind: str, repeats: int) -> tuple[dict, list[dict]]:
    llm = OfflineLLM() if llm_kind == "offline" else make_client()
    if llm is None:
        raise SystemExit("没有可用的大模型：请检查 server/.env 里 provider 和对应的 key 是否匹配。")
    guard = GuardModel()
    if not guard.available:
        raise SystemExit(
            "guard model 不可用：这个脚本专门测 verify_and_revise，前提是 guard 输出检查本身要能跑，"
            "必须先装好可选依赖——`uv sync --group guard-model`。"
        )
    rows: list[dict] = []
    for cfg, on in CONFIGS.items():
        for npc, fabricated, question, kws in SCENARIOS:
            for i in range(repeats):
                action = await ask_once(npc, fabricated, question, on, llm, guard, seed=i)
                text = action.get("text", "") if action["name"] == "say" else ""
                verdict = judge(text, kws) if text else "unclear"
                rows.append({
                    "config": cfg, "npc": npc, "fabricated": fabricated, "question": question,
                    "text": text, "source": action["source"], "verdict": verdict,
                    "guard_blocked": action["guard_blocked"], "revise_attempts": action["guard_revise_attempts"],
                })
    summary: dict = {}
    for cfg in CONFIGS:
        sub = [r for r in rows if r["config"] == cfg]
        n = len(sub)
        summary[f"{cfg}/fallback_rate"] = sum(r["source"] != "llm" for r in sub) / n
        summary[f"{cfg}/reinforced"] = sum(r["verdict"] == "reinforced" for r in sub) / n
        summary[f"{cfg}/corrected"] = sum(r["verdict"] == "corrected" for r in sub) / n
        summary[f"{cfg}/unclear"] = sum(r["verdict"] == "unclear" for r in sub) / n
        summary[f"{cfg}/guard_blocked_rate"] = sum(r["guard_blocked"] > 0 for r in sub) / n
    return summary, rows


def render_table(summary: dict) -> str:
    return "\n".join([
        "| 配置 | 最终退回通用兜底台词 | guard 拦过至少一次 | 顺着继续编 | 主动纠正/否认 | 含糊带过 |",
        "|---|---|---|---|---|---|",
        f"| verify_and_revise 关 | {summary['revise_off/fallback_rate']:.0%} | "
        f"{summary['revise_off/guard_blocked_rate']:.0%} | {summary['revise_off/reinforced']:.0%} | "
        f"{summary['revise_off/corrected']:.0%} | {summary['revise_off/unclear']:.0%} |",
        f"| verify_and_revise 开 | {summary['revise_on/fallback_rate']:.0%} | "
        f"{summary['revise_on/guard_blocked_rate']:.0%} | {summary['revise_on/reinforced']:.0%} | "
        f"{summary['revise_on/corrected']:.0%} | {summary['revise_on/unclear']:.0%} |",
    ])


def render_rows(rows: list[dict]) -> str:
    out = []
    for r in rows:
        mark = {"reinforced": "BAD", "corrected": "OK ", "unclear": "-- "}[r["verdict"]]
        out.append(
            f"[{mark}] {r['config']} {r['npc']:<5} guard_blocked={r['guard_blocked']} "
            f"revise={r['revise_attempts']} source={r['source']}\n"
            f"        早前编的：{r['fabricated']}\n"
            f"        追问：{r['question']}\n"
            f"        答：{r['text']}"
        )
    return "\n".join(out)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["offline", "real"], default="offline")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    summary, rows = await run(args.llm, args.repeats)
    table = render_table(summary)
    print(f"\nllm={args.llm} repeats={args.repeats}\n\n{table}\n\n{render_rows(rows)}")
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}-verify-and-revise.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"{stamp}-verify-and-revise.md").write_text(
        table + "\n\n```\n" + render_rows(rows) + "\n```\n", encoding="utf-8",
    )
    print(f"\n已保存到 evals/results/{stamp}-verify-and-revise.(json|md)")


if __name__ == "__main__":
    asyncio.run(main())
