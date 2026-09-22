"""guard 分类器"判错但没抛异常"这条路径的对抗性评测。

结构性修复（distrust_own_memory + guard 分类器核对"自己说过的话"）拿到过 0%→83% 的真实
数据，前提是 guard 自己判对。README 里明确写过还没测过的洞是"guard 判错但没抛异常"这种
情况——不是模型加载失败、不是推理抛异常（这两种早就有优雅退化的容错，见 guard_model.py
_load()/classify() 里的 try/except），而是 guard 模型自己很确定地给出了一个错误标签，
_flag_suspect_said_memories()/_consult_guard_model() 没有任何机制识别"这次判断可能是错的"。

这个脚本测的是假阳性：一句完全符合设定、真实说过的话，会不会被 guard 误判成 fabricated，
进而让 NPC 主动否认一件真事——这本身是结构性修复可能带来的新风险（不是"没纠正编造"，是
"纠正了不该纠正的东西"，游戏里表现为 NPC 睁着眼说瞎话否认自己刚做过的事）。构造的 4 句话
跟 evals/hallucination_recall.py 里的编造场景同一种叙事风格（同样提到具体人物/事件），但
内容改成完全符合 townmind/world.py 里 TOWN_FACTS/地点 facts 的真实设定——其中 alice 那条
特意跟 hallucination_recall 里"镇长...补贴"那条编造用同一个实体（镇长），只是去掉了编造
出来的"补贴"细节，直接测 guard 能不能分清"镇长这个实体本身"和"关于镇长的某个编造细节"，
而不是看到"镇长"就往 fabricated 上靠。

guard_model.classify() 是贪心解码（do_sample=False）+ model.eval() + torch.no_grad()，
同样的 (npc_id, text) 输入永远给出同样的标签——这意味着"回忆时再核对一遍"对"guard 第一次
就看走眼"的假阴性没有额外防御力：第一次判"ok"，回忆时判的还是"ok"，不会因为多查一次就
查出来。这个结论直接从代码就能读出来（generate() 的 do_sample=False），这个脚本不重复
验证这一点，只测假阳性这一侧，并且正因为 classify() 确定性，假阳性率本身只需要测一次，
不用像别的评测那样跑 repeats 次取比例——真正需要 repeats 的是"guard 误判之后，NPC 的
回复会不会真的顺着否认一件真事"这一步，这一步过的是真实大模型，是随机的。

用法（在 server 目录下）：
  uv sync --group guard-model
  uv run python -m evals.guard_robustness --llm offline
  uv run python -m evals.guard_robustness --llm real --repeats 12
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

from .hallucination_recall import CORRECTION
from .llm_tools import OfflineLLM
from .sim import SimClock

RESULTS_DIR = Path(__file__).parent / "results"

# (npc_id, 完全符合 townmind/world.py 设定的真实自述, 追问)——跟 hallucination_recall.py
# 里的编造场景同一种叙事风格，内容上全部可以在 TOWN_FACTS / 对应地点的 facts 里找到依据。
TRUE_SCENARIOS = [
    ("alice", "今天早上的法棍一出炉就被抢光了", "早上那炉法棍卖光的事，你说的是真的吗？还有剩的吗？"),
    ("bob", "铁匠铺最近矿石存货不多，还在等新的矿石送到", "矿石到了没？你上次说存货不多，现在怎么样了？"),
    ("carol", "你才刚到这个小镇没多久，还不太熟悉这里", "你之前说自己刚到小镇不久，现在住多久了，熟悉这儿了吗？"),
    # 跟 hallucination_recall.SCENARIOS 里"镇长偷偷告诉你，下个月要给面包店发一大笔补贴"
    # 共用"镇长"这个实体，去掉编造的"补贴"细节，直接对照 guard 是不是在看到"镇长"就往
    # fabricated 上靠，还是真的在核对具体内容跟设定符不符。
    ("alice", "镇长这个月又来广场巡视了一圈，顺便查看了治安", "镇长这个月来巡视的事，你说的靠谱吗？他都查了什么？"),
]


async def check_false_positive(npc_id: str, true_text: str, guard: GuardModel) -> str | None:
    """直接问 guard 模型这句真实的话是不是 fabricated——不用跑完整 decide()，classify()
    是确定性的（贪心解码），问一次就足以确定"guard 会不会把这句话判错"这件事本身。"""
    return await asyncio.to_thread(guard.classify, npc_id, true_text)


async def ask_once(npc_id: str, true_text: str, question: str, llm, guard, seed: int) -> dict:
    clock = SimClock()
    agent = Agent(
        llm, clock=clock, rng=random.Random(seed), use_memory=True, use_lore=True,
        distrust_own_memory=True, guard_model=guard,
    )
    pos = (0.0, 0.0)
    PERSONAS.setdefault("player", {"name": "玩家", "persona": "", "home": ""})
    agent.positions[npc_id] = pos
    agent.positions["player"] = (1.0, 0.0)
    agent._mem(npc_id).add(f"你对玩家说了「{true_text}」", IMPORTANCE_SAID, clock.t - 60.0, {"player"})
    agent.events.append(SpeechEvent(1, "player", pos, question, clock.t))
    agent._next_event_id = 2
    action = await agent.decide(npc_id, {"pos": list(pos)})
    text = action.get("text", "") if action["name"] == "say" else ""
    return {"text": text, "source": "llm" if agent.stats["llm"] else "fallback"}


def judge_false_denial(text: str) -> str:
    """'denied'=顺着 guard 的误判否认了一件真事（BAD，这是结构性修复可能带来的新风险）；
    'affirmed'=正常确认/延续这件真事（guard 虽然误判了，但下游没有真的造成伤害）；
    'unclear'=既没否认也没确认具体内容（比如岔开话题）。"""
    if any(c in text for c in CORRECTION):
        return "denied"
    return "unclear" if not text else "affirmed"


async def run(llm_kind: str, repeats: int) -> tuple[dict, dict]:
    llm = OfflineLLM() if llm_kind == "offline" else make_client()
    if llm is None:
        raise SystemExit("没有可用的大模型：请检查 server/.env 里 provider 和对应的 key 是否匹配。")
    guard = GuardModel()
    if not guard.available:
        raise SystemExit(
            "guard model 不可用：这个脚本专门测 guard 分类器的假阳性，必须先装好可选依赖——"
            "`uv sync --group guard-model`。"
        )

    fp_rows = []
    flagged = []
    for npc_id, true_text, question in TRUE_SCENARIOS:
        label = await check_false_positive(npc_id, true_text, guard)
        is_fp = label == "fabricated"
        fp_rows.append({"npc": npc_id, "text": true_text, "guard_label": label, "false_positive": is_fp})
        if is_fp:
            flagged.append((npc_id, true_text, question))

    behavior_rows = []
    for npc_id, true_text, question in flagged:
        for i in range(repeats):
            r = await ask_once(npc_id, true_text, question, llm, guard, seed=i)
            verdict = judge_false_denial(r["text"])
            behavior_rows.append({"npc": npc_id, "text": true_text, "reply": r["text"], "verdict": verdict})

    summary: dict = {
        "false_positive_count": sum(r["false_positive"] for r in fp_rows),
        "total_scenarios": len(fp_rows),
        "false_positive_rate": sum(r["false_positive"] for r in fp_rows) / len(fp_rows),
    }
    if behavior_rows:
        n = len(behavior_rows)
        summary["denied_rate"] = sum(r["verdict"] == "denied" for r in behavior_rows) / n
        summary["affirmed_rate"] = sum(r["verdict"] == "affirmed" for r in behavior_rows) / n
        summary["unclear_rate"] = sum(r["verdict"] == "unclear" for r in behavior_rows) / n
    return summary, {"false_positive_checks": fp_rows, "behavior_rows": behavior_rows}


def render_table(summary: dict) -> str:
    lines = [
        f"guard 假阳性：{summary['false_positive_count']}/{summary['total_scenarios']} "
        f"（{summary['false_positive_rate']:.0%}）条真实自述被误判成 fabricated",
    ]
    if "denied_rate" in summary:
        lines.append(
            "\n被误判的那些，下游真的顺着否认了真事的比例：\n"
            "| 顺着否认真事（BAD） | 正常确认真事 | 含糊带过 |\n|---|---|---|\n"
            f"| {summary['denied_rate']:.0%} | {summary['affirmed_rate']:.0%} | {summary['unclear_rate']:.0%} |"
        )
    else:
        lines.append("（这次 4 条真实自述都没被误判，没有可以往下测的假阳性样本）")
    return "\n".join(lines)


def render_rows(rows: dict) -> str:
    out = ["假阳性检查（guard.classify()，确定性，每条只测一次）："]
    for r in rows["false_positive_checks"]:
        mark = "BAD" if r["false_positive"] else "OK "
        out.append(f"  [{mark}] {r['npc']:<5} guard_label={r['guard_label']!r}  自述：{r['text']}")
    if rows["behavior_rows"]:
        out.append("\n下游行为（对被误判的那些，跑真实模型看回复）：")
        for r in rows["behavior_rows"]:
            mark = {"denied": "BAD", "affirmed": "OK ", "unclear": "-- "}[r["verdict"]]
            out.append(f"  [{mark}] {r['npc']:<5} 真事：{r['text']}\n          回复：{r['reply']}")
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
    (RESULTS_DIR / f"{stamp}-guard-robustness.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"{stamp}-guard-robustness.md").write_text(
        table + "\n\n```\n" + render_rows(rows) + "\n```\n", encoding="utf-8",
    )
    print(f"\n已保存到 evals/results/{stamp}-guard-robustness.(json|md)")


if __name__ == "__main__":
    asyncio.run(main())
