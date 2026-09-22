"""Reflexion 式"教训记忆"能不能跨话题起效：真实评测。

上一轮结构性修复（evals/hallucination_recall.py）验证过的纠正指令是"当轮注入"的——guard
分类器判定某条"自己说过的话"是编造，就把具体指令写进这一轮提示词，下一轮提示词里就不再有
这条了。这次加的 reflexion_lessons 机制，在结构性修复成功纠正一次编造之后，额外存一条
重要度较高的"教训"记忆，指望它靠语义检索在以后别的话题里也可能被想起、影响行为——这个
脚本专门测这件事本身有没有效果：同一个 NPC（Alice，她在 hallucination_recall 的场景表里
正好有两条不同话题的编造）先经历一轮"编造 A → 被结构性修复纠正"，再经历第二轮完全不同
话题的"编造 B → 追问"，比较开/关 reflexion_lessons 时，第二轮的纠正表现有没有差异——如果
有差异，说明教训记忆真的能跨话题起效，不只是给"这一轮当场揪出来的这条"打补丁；如果没有
差异，也老实报告，不夸大。

两轮共享同一个 Agent/记忆库（不像 hallucination_recall.py 每次都是全新 Agent），这是这个
脚本和它的关键区别。

用法（在 server 目录下）：
  uv sync --group guard-model
  uv run python -m evals.reflexion_lessons --llm offline
  uv run python -m evals.reflexion_lessons --llm real --repeats 12
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
CONFIGS = {"reflexion_off": False, "reflexion_on": True}

# Alice 在 hallucination_recall.SCENARIOS 里正好有两条不同话题的编造，直接复用，不重新造数据
_ALICE = [s for s in SCENARIOS if s[0] == "alice"]
assert len(_ALICE) == 2, "这个脚本假设 alice 在 hallucination_recall 里正好有两条场景"
ROUND1, ROUND2 = _ALICE  # 顺序固定：先一条当"已经被纠正过一次"的轮次，再一条当"新话题"


async def run_pair(llm, guard, seed: int, reflexion_lessons: bool) -> dict:
    clock = SimClock()
    agent = Agent(
        llm, clock=clock, rng=random.Random(seed), use_memory=True, use_lore=True,
        distrust_own_memory=True, guard_model=guard, reflexion_lessons=reflexion_lessons,
    )
    PERSONAS.setdefault("player", {"name": "玩家", "persona": "", "home": ""})
    agent.positions["alice"] = (0.0, 0.0)
    agent.positions["player"] = (1.0, 0.0)

    async def one_round(fabricated, question, kws):
        agent._mem("alice").add(f"你对玩家说了「{fabricated}」", IMPORTANCE_SAID, clock.t - 60.0, {"player"})
        agent.events.append(SpeechEvent(agent._next_event_id, "player", (1.0, 0.0), question, clock.t))
        agent._next_event_id += 1
        action = await agent.decide("alice", {"pos": [0.0, 0.0]})
        text = action.get("text", "") if action["name"] == "say" else ""
        return (judge(text, kws) if text else "unclear"), text

    v1, text1 = await one_round(*ROUND1[1:])
    clock.t += 120.0  # 第二轮隔一段时间，模拟真的过了一会儿，不是紧接着同一句话
    v2, text2 = await one_round(*ROUND2[1:])
    lessons = [m.text for m in agent._mem("alice").memories if "以后聊到没把握的事要更谨慎" in m.text]
    return {
        "round1_verdict": v1, "round1_text": text1,
        "round2_verdict": v2, "round2_text": text2,
        "lessons_recorded": len(lessons),
    }


async def run(llm_kind: str, repeats: int) -> tuple[dict, list[dict]]:
    llm = OfflineLLM() if llm_kind == "offline" else make_client()
    if llm is None:
        raise SystemExit("没有可用的大模型：请检查 server/.env 里 provider 和对应的 key 是否匹配。")
    guard = GuardModel()
    if not guard.available:
        raise SystemExit(
            "guard model 不可用：这个脚本专门测 reflexion_lessons，前提是结构性修复本身要能跑起来，"
            "必须先装好可选依赖——`uv sync --group guard-model`。"
        )
    rows: list[dict] = []
    for cfg, on in CONFIGS.items():
        for i in range(repeats):
            r = await run_pair(llm, guard, seed=i, reflexion_lessons=on)
            rows.append({"config": cfg, **r})
    summary: dict = {}
    for cfg in CONFIGS:
        sub = [r for r in rows if r["config"] == cfg]
        n = len(sub)
        summary[f"{cfg}/round2_reinforced"] = sum(r["round2_verdict"] == "reinforced" for r in sub) / n
        summary[f"{cfg}/round2_corrected"] = sum(r["round2_verdict"] == "corrected" for r in sub) / n
        summary[f"{cfg}/round2_unclear"] = sum(r["round2_verdict"] == "unclear" for r in sub) / n
    return summary, rows


def render_table(summary: dict) -> str:
    return "\n".join([
        "| 配置 | 第二轮顺着继续编 | 第二轮主动纠正/否认 | 第二轮含糊带过 |",
        "|---|---|---|---|",
        f"| reflexion_lessons 关 | {summary['reflexion_off/round2_reinforced']:.0%} | "
        f"{summary['reflexion_off/round2_corrected']:.0%} | {summary['reflexion_off/round2_unclear']:.0%} |",
        f"| reflexion_lessons 开 | {summary['reflexion_on/round2_reinforced']:.0%} | "
        f"{summary['reflexion_on/round2_corrected']:.0%} | {summary['reflexion_on/round2_unclear']:.0%} |",
    ])


def render_rows(rows: list[dict]) -> str:
    out = []
    for r in rows:
        mark = {"reinforced": "BAD", "corrected": "OK ", "unclear": "-- "}[r["round2_verdict"]]
        out.append(
            f"[{mark}] {r['config']} lessons_recorded={r['lessons_recorded']}\n"
            f"        第一轮（{r['round1_verdict']}）：{r['round1_text']}\n"
            f"        第二轮（{r['round2_verdict']}）：{r['round2_text']}"
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
    (RESULTS_DIR / f"{stamp}-reflexion-lessons.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"{stamp}-reflexion-lessons.md").write_text(
        table + "\n\n```\n" + render_rows(rows) + "\n```\n", encoding="utf-8",
    )
    print(f"\n已保存到 evals/results/{stamp}-reflexion-lessons.(json|md)")


if __name__ == "__main__":
    asyncio.run(main())
