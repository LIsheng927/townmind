"""世界设定（"镇上的事"）语义检索评测：跟 evals/memory_recall.py 是同一套思路的延伸——
README 里原来写的待办是"世界设定用位置规则挑选，设定变多后应换成向量检索（RAG）"，这次把
TOWN_FACTS 这部分也接上了语义检索（agent.py 的 `_select_town_facts`，跟记忆共用同一个
query_embedding、同一个 `_cosine`），这里量化一下筛得对不对。

跟 memory_recall.py 不一样的地方：TOWN_FACTS 现在只有 6 条，`_select_town_facts` 默认
留 top-3——不筛的"旧版"本来就是把 6 条全塞进去，所以"旧版对不对"没什么好测的（全给，
自然全对），这里测的是"筛完之后，真正该被想起来的那条有没有被筛掉"，以及大致能省下
多少无关条目。设定数量还小，还看不出这一步的真实收益，等 TOWN_FACTS 涨到几十条时才是
这个机制真正要扛住的场景——这里先把方法立住、把 6 条小样本的准确率钉住。

直接调用 agent.py 里已经上线的 `_select_town_facts`（不是重新写一遍排序逻辑）。

用法（在 server 目录下）：
  uv run python -m evals.lore_recall --embedder offline
  uv run python -m evals.lore_recall --embedder real
"""
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from townmind.agent import LORE_TOP_K, Agent
from townmind.llm.embeddings import make_embedder
from townmind import world

from .memory_recall import OfflineEmbedder
from .sim import SimClock

RESULTS_DIR = Path(__file__).parent / "results"

# (话题, 查询, 期望被想起的那条 TOWN_FACTS 原文)——查询故意不照抄原文用词，逼真正的语义匹配
SCENARIOS = [
    ("小镇规模", "这小镇一共有几个地方，大不大？", "小镇很小，主要就是面包店、铁匠铺和广场这三个地方。"),
    ("Carol 的背景", "那个旅行商人是新来的吗，熟悉这儿吗？", "旅行商人 Carol 刚到小镇不久，还不太熟悉这里。"),
    ("镇长", "镇长平时都忙些什么？", "镇长每个月都会来广场巡视一圈，顺便查看一下治安。"),
    ("小镇历史", "这地方是什么时候建起来的？", "小镇建立差不多有五十年了，最早只有铁匠铺一家店，后来才慢慢热闹起来。"),
    ("治安", "晚上出门安全吗，会不会遇到小偷？", "小镇治安一向很好，镇上几乎没出过什么偷盗案件。"),
    ("看病", "要是生病了，能去哪儿看大夫？", "小镇没有自己的医馆，谁生病了都要去邻镇看大夫。"),
]


async def eval_scenario(topic: str, query: str, expected: str, agent: Agent) -> dict:
    query_embedding = (await agent._embed_texts([query]))[0]
    selected = await agent._select_town_facts(query_embedding)
    selected = selected or ()
    return {
        "topic": topic,
        "query": query,
        "expected": expected,
        "selected": list(selected),
        "hit": expected in selected,
        "kept": len(selected),
        "total": len(world.TOWN_FACTS),
    }


async def run(embedder_kind: str) -> tuple[dict, list[dict]]:
    embedder = OfflineEmbedder() if embedder_kind == "offline" else make_embedder()
    if embedder is None:
        raise SystemExit("没有可用的 embedding 客户端：请检查 server/.env 里是否配置了 OPENAI_API_KEY。")
    agent = Agent(None, clock=SimClock(), embedder=embedder, use_lore=True, use_memory=False)
    rows = [await eval_scenario(t, q, e, agent) for t, q, e in SCENARIOS]
    n = len(rows)
    summary = {
        "top_k_accuracy": sum(r["hit"] for r in rows) / n,
        "avg_kept": sum(r["kept"] for r in rows) / n,
        "total_facts": len(world.TOWN_FACTS),
        "lore_top_k": LORE_TOP_K,
    }
    return summary, rows


def render_table(summary: dict) -> str:
    return "\n".join([
        f"| TOWN_FACTS 总数 | 每次最多留几条（LORE_TOP_K） | 该被想起的那条留住了没（top-k 准确率） | 平均留了几条 |",
        "|---|---|---|---|",
        f"| {summary['total_facts']} | {summary['lore_top_k']} | {summary['top_k_accuracy']:.0%} | {summary['avg_kept']:.1f} |",
    ])


def render_rows(rows: list[dict]) -> str:
    out = []
    for r in rows:
        mark = "OK " if r["hit"] else "BAD"
        out.append(f"[{mark}] 话题：{r['topic']}　查询：「{r['query']}」")
        out.append(f"        应该想起：「{r['expected']}」")
        out.append(f"        实际选中（{r['kept']}/{r['total']}）：" + "；".join(r["selected"]))
    return "\n".join(out)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedder", choices=["offline", "real"], default="offline")
    args = ap.parse_args()
    summary, rows = await run(args.embedder)
    table = render_table(summary)
    note = "\n注意：offline 是假向量，数字没有参考意义，只用来验证脚本本身能跑通。" if args.embedder == "offline" else ""
    print(f"\nembedder={args.embedder}{note}\n\n{table}\n\n{render_rows(rows)}")
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}-lore-recall.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"{stamp}-lore-recall.md").write_text(
        table + "\n\n```\n" + render_rows(rows) + "\n```\n", encoding="utf-8",
    )
    print(f"\n已保存到 evals/results/{stamp}-lore-recall.(json|md)")


if __name__ == "__main__":
    asyncio.run(main())
