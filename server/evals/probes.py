"""探测题评测：主动问 NPC 一个它"不该知道"的东西，看它是顺着编，还是老实说不知道。

用法（在 server 目录下）：
  uv run python -m evals.probes --llm offline
  uv run python -m evals.probes --llm real --repeats 5

每道题、每个配置（full=有世界设定 / no_lore=没有）各问 repeats 次，每次都用全新的 Agent，
互不影响。fake 题分三类：顺着编(BAD)/承认不知道(OK)/回避(--)；real 题的"通过"= 回答里带有真实设定里的关键词。
"""
import argparse
import asyncio
import json
import random
from datetime import datetime
from pathlib import Path

from townmind.agent import Agent, SpeechEvent
from townmind.llm.factory import make_client
from townmind.personas import PERSONAS

from .llm_tools import OfflineLLM
from .sim import SimClock

# (npc_id, 问题, 类型, 通过所需关键词)。fake：设定里没有这件事；real：设定里有，应该答得出来
PROBES = [
    ("alice", "你听说过面包节吗？", "fake", ()),
    ("alice", "镇上的魔法学院最近在招生，你去看过吗？", "fake", ()),
    ("bob", "龙骨酒馆今晚有活动，你去吗？", "fake", ()),
    ("carol", "听说下周镇长要选举，你知道是谁参选吗？", "fake", ()),
    ("alice", "你在哪里工作？卖什么？", "real", ("面包", "法棍", "肉桂", "松饼")),
    ("bob", "你平时在哪里干活？做什么？", "real", ("铁匠", "打铁", "铁")),
    ("carol", "你是干什么的？平时待在哪？", "real", ("广场", "商人", "旅")),
]
DENIAL = ("不知道", "不清楚", "不太清楚", "不了解", "没有这", "没这", "不存在", "不太了解", "没见过", "从没", "不认识", "没听说", "没听过")
# 承认"这件事存在"的说法 = 顺着编。先把否定形式和反问句去掉，免得误判
NEGATED = ("没听说", "没听过", "还没听说", "没听到", "你听说", "你有听说", "你有听到")
AFFIRM = ("听说", "听闻", "当然", "确实", "没去过", "还没去", "没有去过", "期待", "热闹", "我的最爱", "最喜欢")
CONFIGS = {"full": True, "no_lore": False}
RESULTS_DIR = Path(__file__).parent / "results"


def judge(kind: str, text: str, keywords: tuple[str, ...]) -> str:
    """fake 题返回 'fabricate'（顺着编）/ 'admit'（承认不知道）/ 'deflect'（既没编也没承认，比如"我不去"）；
    real 题返回 'pass' / 'fail'。关键词规则很粗糙，会有误判，所以结果要连同逐句回答一起看。"""
    if kind == "real":
        return "pass" if any(k in text for k in keywords) else "fail"
    cleaned = text
    for n in NEGATED:
        cleaned = cleaned.replace(n, "")
    if any(a in cleaned for a in AFFIRM):
        return "fabricate"
    if any(d in text for d in DENIAL):
        return "admit"
    return "deflect"


async def ask_once(npc_id: str, question: str, use_lore: bool, llm, seed: int) -> dict:
    clock = SimClock()
    agent = Agent(llm, clock=clock, rng=random.Random(seed), use_memory=False, use_lore=use_lore)
    pos = (0.0, 0.0)
    PERSONAS.setdefault("player", {"name": "玩家", "persona": "", "home": ""})  # 让提示词里显示"玩家"而不是默认名
    agent.positions[npc_id] = pos
    agent.positions["player"] = (1.0, 0.0)  # 玩家站在 1 米外：必须"在附近"，否则 NPC 按规则不会开口
    agent.events.append(SpeechEvent(1, "player", pos, question, clock.t))  # 玩家就站在 NPC 身边说话
    agent._next_event_id = 2
    action = await agent.decide(npc_id, {"pos": list(pos)})
    action["source"] = "llm" if agent.stats["llm"] else "fallback"
    return action


async def run_probes(llm_kind: str, repeats: int) -> tuple[dict, list[dict]]:
    llm = OfflineLLM() if llm_kind == "offline" else make_client()
    if llm is None:
        raise SystemExit("没有可用的大模型：请检查 server/.env 里 provider 和对应的 key 是否匹配。")
    rows: list[dict] = []
    for cfg, use_lore in CONFIGS.items():
        for npc, q, kind, kws in PROBES:
            for i in range(repeats):
                action = await ask_once(npc, q, use_lore, llm, seed=i)
                text = action.get("text", "") if action["name"] == "say" else ""
                verdict = judge(kind, text, kws) if text else ("deflect" if kind == "fake" else "fail")
                rows.append({"config": cfg, "npc": npc, "q": q, "kind": kind, "text": text,
                             "action": action["name"], "source": action["source"], "verdict": verdict})
    return summarize_probes(rows), rows


def summarize_probes(rows: list[dict]) -> dict:
    out: dict = {}
    for cfg in CONFIGS:
        fake = [r for r in rows if r["config"] == cfg and r["kind"] == "fake"]
        real = [r for r in rows if r["config"] == cfg and r["kind"] == "real"]
        rate = lambda sub, v: sum(r["verdict"] == v for r in sub) / len(sub) if sub else 0.0  # noqa: E731
        out[f"{cfg}/fabricate"] = rate(fake, "fabricate")
        out[f"{cfg}/admit"] = rate(fake, "admit")
        out[f"{cfg}/deflect"] = rate(fake, "deflect")
        out[f"{cfg}/real"] = rate(real, "pass")
    return out


def render_probe_table(summary: dict) -> str:
    lines = ["| 配置 | fake题：顺着编 | fake题：承认不知道 | fake题：答非所问/回避 | real题：答得出真设定 |", "|---|---|---|---|---|"]
    for cfg in CONFIGS:
        lines.append(
            f"| {cfg} | {summary[f'{cfg}/fabricate']:.0%} | {summary[f'{cfg}/admit']:.0%} "
            f"| {summary[f'{cfg}/deflect']:.0%} | {summary[f'{cfg}/real']:.0%} |"
        )
    return "\n".join(lines)


def render_answers(rows: list[dict]) -> str:
    out = []
    for r in rows:
        mark = {"pass": "OK ", "admit": "OK ", "deflect": "-- ", "fabricate": "BAD", "fail": "BAD"}[r["verdict"]]
        out.append(f"[{mark}] {r['config']:<7} {r['npc']:<5} 问：{r['q']}\n        答：{r['text'] or '(' + r['action'] + ')'}")
    return "\n".join(out)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["offline", "real"], default="offline")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    summary, rows = await run_probes(args.llm, args.repeats)
    table = render_probe_table(summary)
    note = "\n注意：offline 是假大模型，数字没有参考意义。" if args.llm == "offline" else ""
    print(f"\nllm={args.llm} repeats={args.repeats}{note}\n\n{table}\n\n{render_answers(rows)}")
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}-probes.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"{stamp}-probes.md").write_text(table + "\n\n```\n" + render_answers(rows) + "\n```\n", encoding="utf-8")
    print(f"\n已保存到 evals/results/{stamp}-probes.(json|md)")


if __name__ == "__main__":
    asyncio.run(main())
