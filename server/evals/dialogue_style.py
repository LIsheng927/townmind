"""对话风格评测：README 里承认的局限是"对话仍偏浅，常常复述+反问；Bob 偶尔会说你呢？，
不够沉默寡言"。这次给每个 NPC 加了具体的说话习惯（speech_habits），并把提示词里"形成一来
一回的对话"这句改成明确劝退"复述+反问"套路、鼓励按性格给出实际反应（`expressive_dialogue`
开关，新版默认开，旧版仅用于这里的对比）。

跟其他几个真实模型评测一样，量的是行为有没有真的变，不是猜的：
  反问率：NPC 的回复是不是以"？"结尾——旧提示词明确要求"形成一来一回"，容易导致机械反问
  套话开头率：回复是不是以"是吗/真的吗/有意思"这类敷衍式开场白开头
  平均字数（按角色）：Bob 的"沉默寡言"有没有真的体现在字数上，跟 Alice/Carol 拉开差距

用法（在 server 目录下）：
  uv run python -m evals.dialogue_style --llm offline
  uv run python -m evals.dialogue_style --llm real --repeats 5
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
NPCS = ("alice", "bob", "carol")

# 都是不涉及镇上具体设定的闲聊，避免跟"防编造"那条规则的行为混在一起
HEARD_LINES = [
    "今天天气真好啊。",
    "你最近还好吗？",
    "我觉得你这活儿干得不错。",
    "最近感觉有点无聊啊。",
    "你干这行干了很久了吧？",
    "今天感觉怎么样？",
]

# 敷衍式开场白：先复述/附和一下，再展开——这正是想弱化的套路。
# 第一版只列了"是吗/真的吗"这类疑问式附和，真实数据里最常见的其实是"是啊/是呀"这种
# 陈述式附和（比如听到"今天天气真好啊"就回"是啊，今天天气真不错..."，本质是同一种套路，
# 只是语气词不同）——这是跑完真实评测之后才发现的列表漏洞，补上了。
FILLERS = (
    "是吗", "真的吗", "这样啊", "原来如此", "有意思", "真棒", "不错哦",
    "哦，是这样", "嗯，是这样", "确实", "是啊", "是呀", "是的", "没错",
)


async def ask_once(npc_id: str, line: str, expressive: bool, llm, seed: int) -> dict:
    clock = SimClock()
    agent = Agent(llm, clock=clock, rng=random.Random(seed), use_memory=False, use_lore=True, expressive_dialogue=expressive)
    agent.positions[npc_id] = (0.0, 0.0)
    agent.hear_player(line, [1.0, 0.0])
    action = await agent.decide(npc_id, {"pos": [0.0, 0.0]})
    text = action.get("text", "") if action["name"] == "say" else ""
    return {
        "npc_id": npc_id,
        "line": line,
        "expressive": expressive,
        "action": action["name"],
        "text": text,
        "is_say": action["name"] == "say",
        "ends_question": bool(text) and text.rstrip().endswith(("？", "?")),
        "starts_filler": bool(text) and text.lstrip().startswith(FILLERS),
        "length": len(text) if text else None,
    }


async def run(llm_kind: str, repeats: int) -> tuple[dict, list[dict]]:
    llm = OfflineLLM() if llm_kind == "offline" else make_client()
    if llm is None:
        raise SystemExit("没有可用的大模型：请检查 server/.env 里 provider 和对应的 key 是否匹配。")
    rows: list[dict] = []
    for npc_id in NPCS:
        for line in HEARD_LINES:
            for expressive in (False, True):
                for i in range(repeats):
                    rows.append(await ask_once(npc_id, line, expressive, llm, seed=i))
    return summarize(rows), rows


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for npc_id in NPCS:
        out[npc_id] = {}
        for expressive in (False, True):
            sub = [r for r in rows if r["npc_id"] == npc_id and r["expressive"] == expressive]
            says = [r for r in sub if r["is_say"]]
            out[npc_id]["expressive" if expressive else "classic"] = {
                "say_rate": len(says) / len(sub) if sub else 0.0,
                "question_rate": sum(r["ends_question"] for r in says) / len(says) if says else 0.0,
                "filler_rate": sum(r["starts_filler"] for r in says) / len(says) if says else 0.0,
                "avg_length": sum(r["length"] for r in says) / len(says) if says else 0.0,
                "n": len(sub),
            }
    return out


def render_table(summary: dict) -> str:
    lines = [
        "| NPC | 版本 | say 比例 | 反问率 | 套话开头率 | 平均字数 |",
        "|---|---|---|---|---|---|",
    ]
    labels = {"classic": "旧（形成一来一回）", "expressive": "新（弱化套路+说话习惯）"}
    for npc_id in NPCS:
        for key in ("classic", "expressive"):
            s = summary[npc_id][key]
            lines.append(
                f"| {npc_id} | {labels[key]}（n={s['n']}） | {s['say_rate']:.0%} | "
                f"{s['question_rate']:.0%} | {s['filler_rate']:.0%} | {s['avg_length']:.1f} |"
            )
    return "\n".join(lines)


def render_rows(rows: list[dict]) -> str:
    out = []
    for r in rows:
        if not r["is_say"]:
            out.append(f"[{r['npc_id']}/{'新' if r['expressive'] else '旧'}] 听到「{r['line']}」-> ({r['action']})")
            continue
        marks = []
        if r["ends_question"]:
            marks.append("反问")
        if r["starts_filler"]:
            marks.append("套话开头")
        tag = f"[{'/'.join(marks)}]" if marks else ""
        out.append(f"[{r['npc_id']}/{'新' if r['expressive'] else '旧'}] 听到「{r['line']}」-> 「{r['text']}」{tag}")
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
    (RESULTS_DIR / f"{stamp}-dialogue-style.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"{stamp}-dialogue-style.md").write_text(
        table + "\n\n```\n" + render_rows(rows) + "\n```\n", encoding="utf-8",
    )
    print(f"\n已保存到 evals/results/{stamp}-dialogue-style.(json|md)")


if __name__ == "__main__":
    asyncio.run(main())
