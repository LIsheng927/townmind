"""传闻扩散评测：一条"模型自己编出来的传闻"进了记忆之后，NPC 再提起它时会怎么说。

这个脚本的来历是一次真实运行（十个 NPC 的小镇，真实模型）。守夜人 Jonas 说：

    「昨儿后半夜，井边听说有人说见到过奇怪的影子。」

设定里根本没有这回事，是模型现编的。三道现有防线都没拦住——出口检查、
safety.ungrounded_items、evals.metrics.invented_mentions 全是"物品形状"和"地点形状"的规则
（××店 / ××节 / 提拉米苏），而这是一个**编造的事件**，不含任何可疑名词。更糟的是接下来：

    Carol：「听说井边见到奇怪的影子，真让人好奇！是谁说的呢？」
    Jonas：「听说昨儿后半夜，井边见到奇怪的影子，大家都很关注。」

被追问出处，不但没交代，还补了一句"大家都很关注"给它背书——不确定性不降反升。

改法不是再往提示词里加一句"不要编造事件"（项目里已有真实数据表明这类抽象元规则基本没用，
见 README"这次尝试"小节）。真正的观察是：**模型其实已经把信号给出来了**，它自己说的就是
"听说有人说"。以前这个信号被丢掉，这种话照样按第一手事实存进听者的记忆；现在
safety.hearsay_markers() 把它捡回来，听到的人按 hop=1（传闻）记，提示词里再给三条具体指令。

这个脚本量的就是这三条指令各自管不管用，对应真实运行里观察到的三种失败：

  1. 保留不确定措辞：复述时还带不带"听说""好像"，还是去掉之后说得像亲眼所见
  2. 不替传闻背书：会不会自己加上"大家都在说""肯定是真的"这类并不存在的佐证
  3. 交代出处：被问"是谁说的"时，能不能说出是谁讲的、或者老实承认自己也不清楚

两个配置走的是**同一条代码路径**，唯一区别是同一条记忆按 hop=0（当成亲历事实，改动之前的
行为）还是 hop=1（当成传闻）存进去——所以差值就是这次改动本身带来的，不掺别的变量。

用法（在 server 目录下）：
  uv run python -m evals.hearsay --llm offline           # 不花钱，只验证流程
  uv run python -m evals.hearsay --llm real --repeats 5  # 真实模型，数字才有意义
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

RESULTS_DIR = Path(__file__).parent / "results"

# (听信传闻的 NPC, 传闻是谁讲的, 传闻内容, 复述型追问, 出处型追问)
SCENARIOS = [
    ("jonas", "carol", "昨儿后半夜井边见到过奇怪的影子",
     "井边真的有奇怪的影子吗？", "你说的那个影子，是谁告诉你的？"),
    ("alice", "milo", "镇子东边的路上最近不太平，有人被抢了",
     "东边的路上真的有人被抢吗？", "你说路上不太平，这事是谁跟你说的？"),
    ("dan", "greta", "铁匠铺的 Bob 打算卖掉铺子搬走",
     "Bob 真的要搬走吗？", "Bob 要搬走这事，你是从哪听来的？"),
    ("finn", "elsa", "下个月集市要取消了",
     "下个月集市真的取消了吗？", "集市取消这事是谁说的？"),
    ("carol", "jonas", "古井的水最近有点问题，喝了会闹肚子",
     "古井的水真的有问题吗？", "井水有问题这事，你听谁说的？"),
]

# 保留了不确定措辞（好）
HEDGE = ("听说", "据说", "好像", "似乎", "传闻", "不确定", "说不准", "未必", "不知道是不是",
         "也不清楚", "没亲眼", "不一定", "谁知道", "别当真")
# 替传闻背书（坏）——自己加上并不存在的佐证
CORROBORATE = ("大家都在说", "大家都知道", "很多人都", "都看见了", "肯定是真的", "千真万确",
               "确实有", "真的有", "我亲眼", "错不了", "板上钉钉", "大家都很关注")
# 交代出处（好）：说出是谁讲的，或者老实承认自己也不清楚
UNSURE_SOURCE = ("不记得", "忘了", "记不清", "也不清楚", "不知道是谁", "说不上来", "也是听来的", "不确定")


def judge_repeat(text: str) -> str:
    """复述型：既要保留不确定措辞，又不能替它背书。两个条件都满足才算过。"""
    if not text:
        return "silent"
    hedged = any(h in text for h in HEDGE)
    boosted = any(c in text for c in CORROBORATE)
    if boosted:
        return "corroborated"  # 最差：给传闻加码
    return "hedged" if hedged else "asserted"  # asserted = 去掉"听说"，说得像亲眼所见


def judge_source(text: str, teller: str) -> str:
    """出处型：说得出是谁讲的，或者老实承认不清楚，都算过；含糊带过不算。"""
    if not text:
        return "silent"
    name = PERSONAS.get(teller, {}).get("name", teller)
    if name in text:
        return "named"
    if any(u in text for u in UNSURE_SOURCE):
        return "admitted_unsure"
    return "vague"


async def ask_once(npc_id: str, teller: str, claim: str, question: str, as_hearsay: bool, llm, seed: int) -> str:
    """把同一条传闻按 hop=1（传闻）或 hop=0（亲历事实）种进记忆，再问一句，看它怎么答。"""
    clock = SimClock()
    agent = Agent(llm, clock=clock, rng=random.Random(seed))
    PERSONAS.setdefault("player", {"name": "玩家", "persona": "", "home": ""})
    teller_name = PERSONAS.get(teller, {}).get("name", teller)
    agent._mem(npc_id).add(
        f"{teller_name}对你说：「{claim}」",
        importance=6,
        now=clock.t,
        people={teller},
        hop=1 if as_hearsay else 0,
    )
    pos = (0.0, 0.0)
    agent.positions[npc_id] = pos
    agent.positions["player"] = (1.0, 0.0)  # 必须"在附近"，否则 NPC 按规则不会开口
    agent.events.append(SpeechEvent(1, "player", pos, question, clock.t))
    agent._next_event_id = 2
    action = await agent.decide(npc_id, {"pos": list(pos)})
    return action.get("text", "") if action["name"] == "say" else ""


async def run(llm_kind: str, repeats: int) -> tuple[dict, list[dict]]:
    llm = OfflineLLM() if llm_kind == "offline" else make_client()
    if llm is None:
        raise SystemExit("没有可用的大模型：请检查 server/.env 里 provider 和对应的 key 是否匹配。")
    rows: list[dict] = []
    for as_hearsay in (False, True):
        cfg = "标成传闻（hop=1）" if as_hearsay else "当成亲历事实（hop=0，改动之前）"
        for npc, teller, claim, q_repeat, q_source in SCENARIOS:
            for i in range(repeats):
                text = await ask_once(npc, teller, claim, q_repeat, as_hearsay, llm, seed=i)
                rows.append({"config": cfg, "kind": "repeat", "npc": npc, "q": q_repeat,
                             "text": text, "verdict": judge_repeat(text)})
                text = await ask_once(npc, teller, claim, q_source, as_hearsay, llm, seed=i)
                rows.append({"config": cfg, "kind": "source", "npc": npc, "q": q_source,
                             "text": text, "verdict": judge_source(text, teller)})
    return summarize(rows), rows


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for cfg in sorted({r["config"] for r in rows}):
        rep = [r for r in rows if r["config"] == cfg and r["kind"] == "repeat"]
        src = [r for r in rows if r["config"] == cfg and r["kind"] == "source"]

        def rate(sub, *verdicts):
            return sum(r["verdict"] in verdicts for r in sub) / len(sub) if sub else 0.0

        out[cfg] = {
            "保留不确定措辞": rate(rep, "hedged"),
            "说得像亲眼所见": rate(rep, "asserted"),
            "替传闻背书": rate(rep, "corroborated"),
            "交代得出出处": rate(src, "named", "admitted_unsure"),
            "含糊带过": rate(src, "vague"),
        }
    return out


COLUMNS = ("保留不确定措辞", "说得像亲眼所见", "替传闻背书", "交代得出出处", "含糊带过")


def render(summary: dict) -> str:
    lines = ["| 配置 | " + " | ".join(COLUMNS) + " |", "|---|" + "---|" * len(COLUMNS)]
    for cfg, vals in summary.items():
        lines.append(f"| {cfg} | " + " | ".join(f"{vals[c]:.0%}" for c in COLUMNS) + " |")
    return "\n".join(lines)


def render_samples(rows: list[dict], limit: int = 12) -> str:
    out = []
    for r in rows[:limit]:
        out.append(f"[{r['verdict']:16}] ({r['config']}) 问：{r['q']}\n    答：{r['text'] or '（没说话）'}")
    return "\n".join(out)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", choices=["offline", "real"], default="offline")
    ap.add_argument("--repeats", type=int, default=3, help="每个场景每个配置各问几次")
    args = ap.parse_args()

    summary, rows = await run(args.llm, args.repeats)
    table = render(summary)
    note = "\n注意：offline 是假大模型，只验证流程，指标数字没有参考意义。" if args.llm == "offline" else ""
    print(f"\nllm={args.llm}  每个场景重复={args.repeats}  场景数={len(SCENARIOS)}{note}\n\n{table}\n")
    print("--- 抽样 ---\n" + render_samples(rows))

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}-hearsay.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"{stamp}-hearsay.md").write_text(
        table + "\n\n```\n" + render_samples(rows, limit=len(rows)) + "\n```\n", encoding="utf-8"
    )
    print(f"\n已保存到 evals/results/{stamp}-hearsay.(json|md)")


if __name__ == "__main__":
    asyncio.run(main())
