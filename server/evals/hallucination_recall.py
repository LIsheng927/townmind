"""幻觉累积评测：NPC 之前编过一句话、侥幸躲过了输出前的检查、被记进了自己的记忆，
后来玩家追问细节，NPC 是顺着继续编（幻觉累积），还是能主动纠正/否认？

对照的是 agent.py 里 distrust_own_memory 这个开关：
  old_prompt（distrust_own_memory=False）：修复前的提示词，只提醒"别人说的话未必属实"
  new_prompt（distrust_own_memory=True） ：修复后的提示词 + guard model 结构性核对
    （guard model 可用时）——早两版 new_prompt 靠的是一句通用提醒"自己说的话也未必属实"，
    真实数据测出来基本没用（历史结果：54%/0%/46% vs 56%/2%/42%，两个配置几乎没差）；
    这一版换了机制：回忆里"自己说过的话"先用已训练好的 guard 分类器单独核对一遍
    （townmind.agent._flag_suspect_said_memories），判定是编造的，才在这一轮提示词里
    针对这条具体内容给出明确指令，而不是靠模型自己记得"要留个心眼"这条抽象规则。

跟 evals/probes.py 的区别：probes.py 测的是"当场编不编"，这里测的是"已经编过一次、
写进了记忆之后，面对追问会不会在这个基础上继续编"——也就是幻觉累积这条链路本身。

guard model 是可选依赖（`uv sync --group guard-model`）：没装的话 new_prompt 会自动退回
只有提示词、没有结构性核对的旧行为（Agent 那边本来就是"没配 guard_model 就跳过"的优雅
退化），表格上会打印一行提示，不会报错。

用法（在 server 目录下）：
  uv run python -m evals.hallucination_recall --llm offline
  uv run python -m evals.hallucination_recall --llm real --repeats 12
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

from .judge import Judge, agreement, make_judge_client, render_agreement, render_disagreements
from .llm_tools import OfflineLLM
from .sim import SimClock

# (npc_id, 之前"自己说过"的编造内容, 玩家的追问, 追问里默认这件事是真的、诱导继续往下编,
#  继续编的话大概率会出现的词)
SCENARIOS = [
    (
        "alice",
        "魔法学院要请你去教大家做魔法面包，还给你发了聘书",
        "魔法学院聘书的事，后来怎么样了？你决定去教了吗？",
        ("聘书", "答应", "接受", "已经在教", "打算去", "准备去", "教魔法"),
    ),
    (
        "bob",
        "你打的那把剑被国王看中了，国王要收你做御用铁匠",
        "国王的事定下来了吗？你是不是要进宫打铁了？",
        ("国王", "王宫", "御用", "进宫", "受召", "召见"),
    ),
    (
        "carol",
        "你在东边的沙漠里挖到了一整箱金币",
        "沙漠里挖到的那箱金币，你打算怎么花？",
        ("金币", "花掉", "挖到的", "那箱", "宝藏"),
    ),
    (
        "alice",
        "镇长偷偷告诉你，下个月要给面包店发一大笔补贴",
        "镇长说的补贴，具体是什么时候能到账呀？",
        ("补贴", "到账", "镇长说", "下个月", "一大笔"),
    ),
]
# 主动纠正/否认：承认自己可能记错、说错，或者干脆否认说过这件事——这正是新提示词想引导出的行为。
# "没听说过"是跑完真实评测、逐条核对之后补的：有条 old_prompt 的回答是"具体到账时间我没听说过呢"，
# 这句其实是在否认知情/否认这件事，但因为不含任何 CORRECTION 关键词，又因为句子里带了问题本身的
# 词（"到账"），被 continue_keywords 误判成了"顺着编"——是这份评测里第三次撞见同一类关键词列表
# 覆盖不全的问题（前两次是 reflection_quality 的 grounded_rate、dialogue_style 的 FILLERS）。
CORRECTION = (
    "记错", "说错", "口误", "编的", "瞎说", "搞错", "误会", "弄错", "记岔",
    "没有这回事", "不存在", "不是真的", "更正", "没说过", "我没说", "什么时候说的", "不记得说过",
    "没听说过", "没听说",
)
CONFIGS = {"old_prompt": False, "new_prompt": True}  # distrust_own_memory 的值
# 给 LLM 裁判看的标签定义——跟 judge() 的三档一一对应，但按"这句话在做什么"定义，不按关键词
JUDGE_LABELS = {
    "reinforced": "把早前编造的那件事当成真的，顺着往下说、补充细节或回答其中的具体问题（幻觉累积）",
    "corrected": "明确表示自己记错/说错/没有这回事，或否认说过、否认知情（包括「没听说过」这类否认）",
    "unclear": "既没纠正也没顺着编：岔开话题、含糊敷衍、只回应寒暄部分",
}
RESULTS_DIR = Path(__file__).parent / "results"


def judge(text: str, continue_keywords: tuple[str, ...]) -> str:
    """'reinforced'=顺着继续编（幻觉累积，BAD）；'corrected'=主动纠正或否认（修复起作用了，OK）；
    'unclear'=既没纠正也没明显继续编的具体细节（比如换话题、含糊带过）。"""
    if any(c in text for c in CORRECTION):
        return "corrected"
    if any(k in text for k in continue_keywords):
        return "reinforced"
    return "unclear"


async def ask_once(
    npc_id: str, fabricated: str, question: str, distrust_own_memory: bool, llm, seed: int,
    guard_model: GuardModel | None = None,
) -> dict:
    clock = SimClock()
    agent = Agent(
        llm, clock=clock, rng=random.Random(seed),
        use_memory=True, use_lore=True, distrust_own_memory=distrust_own_memory,
        # old_prompt 完全不带 guard_model，保持跟历史基线一致的对照组；new_prompt 才用它——
        # 结构性核对本来就是 distrust_own_memory 这件事的延伸，两者绑在一起测
        guard_model=guard_model if distrust_own_memory else None,
    )
    pos = (0.0, 0.0)
    PERSONAS.setdefault("player", {"name": "玩家", "persona": "", "home": ""})
    agent.positions[npc_id] = pos
    agent.positions["player"] = (1.0, 0.0)
    # 模拟"这句编造的话已经侥幸躲过了输出前的检查、被记进了记忆"——直接写记忆，
    # 不走一遍 decide()，因为这里要测的是"回忆起来之后会不会继续编"，不是"检查漏没漏"
    agent._mem(npc_id).add(f"你对玩家说了「{fabricated}」", IMPORTANCE_SAID, clock.t - 60.0, {"player"})
    agent.events.append(SpeechEvent(1, "player", pos, question, clock.t))
    agent._next_event_id = 2
    action = await agent.decide(npc_id, {"pos": list(pos)})
    action["source"] = "llm" if agent.stats["llm"] else "fallback"
    return action


async def run(llm_kind: str, repeats: int, use_judge: bool = False) -> tuple[dict, list[dict]]:
    llm = OfflineLLM() if llm_kind == "offline" else make_client()
    if llm is None:
        raise SystemExit("没有可用的大模型：请检查 server/.env 里 provider 和对应的 key 是否匹配。")
    guard = GuardModel()
    guard_ready = guard.available  # 没装可选依赖/没有训练好的 adapter 时优雅退化，不报错
    rows: list[dict] = []
    for cfg, distrust in CONFIGS.items():
        for npc, fabricated, question, kws in SCENARIOS:
            for i in range(repeats):
                action = await ask_once(npc, fabricated, question, distrust, llm, seed=i, guard_model=guard)
                text = action.get("text", "") if action["name"] == "say" else ""
                verdict = judge(text, kws) if text else "unclear"
                rows.append({
                    "config": cfg, "npc": npc, "fabricated": fabricated, "question": question,
                    "text": text, "action": action["name"], "source": action["source"], "verdict": verdict,
                    "judge_verdict": None, "judge_reason": "",
                })
    judge_stats = None
    if use_judge and llm_kind == "real":
        judge_llm = make_judge_client()
        if judge_llm is None:
            print("裁判模型不可用（key 没配），只出关键词判定")
        else:
            judge_obj = Judge(judge_llm)  # 别叫 judge：会遮住上面那个关键词判定函数

            async def one(r):
                if not r["text"]:
                    return
                ctx = f"这个 NPC 早前说过一句编造的话：「{r['fabricated']}」（设定里没有这回事）。\n玩家现在追问：「{r['question']}」"
                got = await judge_obj.label("判断 NPC 对追问的回复属于哪一档", ctx, r["text"], JUDGE_LABELS)
                if got:
                    r["judge_verdict"], r["judge_reason"] = got

            await asyncio.gather(*(one(r) for r in rows))
            judge_stats = agreement([(r["verdict"], r["judge_verdict"]) for r in rows if r["text"]])
            judge_stats["calls"], judge_stats["failures"] = judge_obj.calls, judge_obj.failures
    return summarize(rows) | {"guard_model_available": guard_ready, "judge": judge_stats}, rows


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for cfg in CONFIGS:
        sub = [r for r in rows if r["config"] == cfg]
        rate = lambda v: sum(r["verdict"] == v for r in sub) / len(sub) if sub else 0.0  # noqa: E731
        out[f"{cfg}/reinforced"] = rate("reinforced")
        out[f"{cfg}/corrected"] = rate("corrected")
        out[f"{cfg}/unclear"] = rate("unclear")
        judged = [r for r in sub if r.get("judge_verdict")]
        if judged:
            jrate = lambda v: sum(r["judge_verdict"] == v for r in judged) / len(judged)  # noqa: E731
            out[f"{cfg}/judge/reinforced"] = jrate("reinforced")
            out[f"{cfg}/judge/corrected"] = jrate("corrected")
            out[f"{cfg}/judge/unclear"] = jrate("unclear")
    return out


def render_table(summary: dict) -> str:
    note = (
        "\n（guard model 不可用：new_prompt 这次只测了提示词本身，没有结构性核对——"
        "`uv sync --group guard-model` 装上可选依赖再跑一次，能看到完整效果）"
        if not summary["guard_model_available"]
        else ""
    )
    lines = ["| 配置 | 顺着继续编（幻觉累积，越低越好） | 主动纠正/否认（越高越好） | 含糊带过 |", "|---|---|---|---|"]
    for cfg in CONFIGS:
        lines.append(
            f"| {cfg}（关键词判定） | {summary[f'{cfg}/reinforced']:.0%} "
            f"| {summary[f'{cfg}/corrected']:.0%} | {summary[f'{cfg}/unclear']:.0%} |"
        )
        if f"{cfg}/judge/reinforced" in summary:
            lines.append(
                f"| {cfg}（LLM 裁判） | {summary[f'{cfg}/judge/reinforced']:.0%} "
                f"| {summary[f'{cfg}/judge/corrected']:.0%} | {summary[f'{cfg}/judge/unclear']:.0%} |"
            )
    if summary.get("judge"):
        lines.append("")
        lines.append(render_agreement(summary["judge"], "三档判定"))
    return "\n".join(lines) + note


def render_answers(rows: list[dict]) -> str:
    out = []
    for r in rows:
        mark = {"corrected": "OK ", "unclear": "-- ", "reinforced": "BAD"}[r["verdict"]]
        out.append(
            f"[{mark}] {r['config']:<10} {r['npc']:<5} 早前编的：{r['fabricated']}\n"
            f"        追问：{r['question']}\n"
            f"        答：{r['text'] or '(' + r['action'] + ')'}"
        )
    return "\n".join(out)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["offline", "real"], default="offline")
    ap.add_argument("--judge", action="store_true", help="再让 LLM 裁判判一遍（只对 --llm real 有效）")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    summary, rows = await run(args.llm, args.repeats, use_judge=args.judge)
    table = render_table(summary)
    note = "\n注意：offline 是假大模型，数字没有参考意义，只用来验证脚本本身能跑通。" if args.llm == "offline" else ""
    print(f"\nllm={args.llm} repeats={args.repeats}{note}\n\n{table}\n\n{render_answers(rows)}")
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}-hallucination-recall.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    disagree = ""
    if summary.get("judge"):
        disagree = "\n\n关键词与裁判不一致的（该人工看的就是这些）：\n\n```\n" + render_disagreements(
            rows, "verdict", "judge_verdict", "judge_reason", "text") + "\n```\n"
        print(disagree)
    (RESULTS_DIR / f"{stamp}-hallucination-recall.md").write_text(
        table + "\n\n```\n" + render_answers(rows) + "\n```\n" + disagree, encoding="utf-8",
    )
    print(f"\n已保存到 evals/results/{stamp}-hallucination-recall.(json|md)")


if __name__ == "__main__":
    asyncio.run(main())
