"""语义检索准确率评测：MemoryStore.recall() 的"相关度"这一项，从"这条记忆涉及的人此刻
在不在场"（旧公式）换成了"跟当前情境语义上有多像"（新公式，余弦相似度）之后，
到底能不能真的把该被想起来的那条记忆排到前面？

每个场景里有 4 条记忆，重要度、时间、涉及的人全部设成完全一样——这是故意的：
这样两个公式唯一的区别就只剩"相关度"这一项，干净地隔离出这次改动到底起不起作用，
不会被"反正这条记忆本来就更新/更重要"这种巧合混进来。4 条里只有 1 条真的跟场景的
"当前情境"是一个话题，其余 3 条是无关的干扰项，且"正确答案"在列表里的位置故意轮换，
不让旧公式因为"永远选第一条"而蒙对。

旧公式在这种设定下完全没法区分 4 条记忆（人都一样、分都一样），Python 的排序是稳定的，
所以它会不管问的是什么、永远选同一条——这正是新公式要解决的真实局限，不是编出来的稻草人。

用法（在 server 目录下）：
  uv run python -m evals.memory_recall --embedder offline
  uv run python -m evals.memory_recall --embedder real
"""
import argparse
import asyncio
import json
import random
import zlib
from datetime import datetime
from pathlib import Path

from townmind.llm.embeddings import make_embedder
from townmind.memory import MemoryStore

RESULTS_DIR = Path(__file__).parent / "results"
NOW = 1000.0
INVOLVED = {"bob"}  # 4 条记忆全部挂在同一个人身上，旧公式的"相关度"因此对它们完全没有区分度

# (话题, 当前情境的查询文本, 4 条候选记忆, 正确答案在候选里的下标)
# 正确答案的位置刻意轮换（2/0/3/1/2/0），不让"永远选第一条"的旧公式蒙对太多次
SCENARIOS = [
    {
        "topic": "剑的价格",
        "query": "这把剑到底能卖多少钱？",
        "candidates": [
            "Bob 说他昨晚没睡好，一直在想集市摆摊的事",
            "Bob 说他很喜欢镇上新开的面包店，常去买法棍",
            "Bob 说他打的这把剑品质很好，能卖上好价钱",
            "Bob 说他觉得最近天气忽冷忽热，容易感冒",
        ],
        "correct_idx": 2,
    },
    {
        "topic": "面包和烘焙",
        "query": "法棍好吃吗，要不要再来一个？",
        "candidates": [
            "Bob 说他今天尝了一口刚出炉的法棍，很喜欢那个味道",
            "Bob 说他这个月要打不少农具，铁匠铺挺忙的",
            "Bob 说他准备周末去广场看看热闹",
            "Bob 说他攒了点钱，想买一把新剑防身",
        ],
        "correct_idx": 0,
    },
    {
        "topic": "天气",
        "query": "今天是不是要下雨了？",
        "candidates": [
            "Bob 说他昨天去广场逛了逛，人挺多的",
            "Bob 说他手上这把剑是照着老图纸打的",
            "Bob 说他今天吃了两个肉桂卷当早饭",
            "Bob 说他觉得云压得很低，八成要下雨",
        ],
        "correct_idx": 3,
    },
    {
        "topic": "集市和广场",
        "query": "周末的集市你打算去逛逛吗？",
        "candidates": [
            "Bob 说他打铁的手艺是跟父亲学的",
            "Bob 说他听说这周末广场有集市，想去看看",
            "Bob 说他最近总觉得腰有点酸",
            "Bob 说他家里面粉快吃完了",
        ],
        "correct_idx": 1,
    },
    {
        "topic": "旅行经历",
        "query": "你以前是不是去过很远的地方？",
        "candidates": [
            "Bob 说他最近铁矿进货的价格涨了",
            "Bob 说他今天心情不错，多打了两把镰刀",
            "Bob 说他年轻时去过很远的地方做生意",
            "Bob 说他不太爱吃甜的东西",
        ],
        "correct_idx": 2,
    },
    {
        "topic": "打铁手艺",
        "query": "你这门打铁的手艺是怎么学的？",
        "candidates": [
            "Bob 说他这手打铁的功夫练了十几年了",
            "Bob 说他昨天路过面包店，闻着味道挺香",
            "Bob 说他不太喜欢人多的集市",
            "Bob 说他觉得今天风有点大",
        ],
        "correct_idx": 0,
    },
]


class OfflineEmbedder:
    """确定性的假 embedding：不联网、不花钱，只用来验证脚本本身能跑通。

    ⚠ 向量是按文字的 hash 生成的，不含任何真实语义，这个模式下的准确率数字没有参考意义
    （大概率两个公式都接近瞎蒙的水平）。
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for t in texts:
            rng = random.Random(zlib.crc32(t.encode("utf-8")))
            vectors.append([rng.uniform(-1.0, 1.0) for _ in range(16)])
        return vectors


async def eval_scenario(scenario: dict, embedder) -> dict:
    texts = scenario["candidates"]
    correct = texts[scenario["correct_idx"]]

    store = MemoryStore()
    for t in texts:
        # 重要度、时间、涉及的人全部一样：唯一能造成排序差异的只剩"相关度"这一项
        store.add(t, importance=5, now=NOW, people=INVOLVED)

    old_top1 = store.recall(INVOLVED, NOW, k=1)[0].text
    old_top2 = [m.text for m in store.recall(INVOLVED, NOW, k=2)]

    vectors = await embedder.embed([*texts, scenario["query"]])
    mem_vecs, query_vec = vectors[:-1], vectors[-1]
    for m, v in zip(store.memories, mem_vecs):
        m.embedding = tuple(v)  # 只测检索公式本身；写入链路（_remember 批量算 embedding）已经在 test_agent.py 里测过

    new_top1 = store.recall(INVOLVED, NOW, k=1, query_embedding=query_vec)[0].text
    new_top2 = [m.text for m in store.recall(INVOLVED, NOW, k=2, query_embedding=query_vec)]

    return {
        "topic": scenario["topic"],
        "query": scenario["query"],
        "correct": correct,
        "old_top1_pick": old_top1,
        "old_top1_correct": old_top1 == correct,
        "old_top2_correct": correct in old_top2,
        "new_top1_pick": new_top1,
        "new_top1_correct": new_top1 == correct,
        "new_top2_correct": correct in new_top2,
    }


async def run(embedder_kind: str) -> tuple[dict, list[dict]]:
    if embedder_kind == "offline":
        embedder = OfflineEmbedder()
    else:
        embedder = make_embedder()
        if embedder is None:
            raise SystemExit("没有可用的 embedding 客户端：请检查 server/.env 里是否配置了 OPENAI_API_KEY。")
    rows = [await eval_scenario(s, embedder) for s in SCENARIOS]
    n = len(rows)
    summary = {
        "old_top1_accuracy": sum(r["old_top1_correct"] for r in rows) / n,
        "old_top2_accuracy": sum(r["old_top2_correct"] for r in rows) / n,
        "new_top1_accuracy": sum(r["new_top1_correct"] for r in rows) / n,
        "new_top2_accuracy": sum(r["new_top2_correct"] for r in rows) / n,
    }
    return summary, rows


def render_table(summary: dict) -> str:
    lines = [
        "| 公式 | Top-1 准确率（该条排第一） | Top-2 准确率（该条进前二） |",
        "|---|---|---|",
        f"| 旧公式（认不认人） | {summary['old_top1_accuracy']:.0%} | {summary['old_top2_accuracy']:.0%} |",
        f"| 新公式（语义相似度） | {summary['new_top1_accuracy']:.0%} | {summary['new_top2_accuracy']:.0%} |",
    ]
    return "\n".join(lines)


def render_rows(rows: list[dict]) -> str:
    out = []
    for r in rows:
        old_mark = "OK " if r["old_top1_correct"] else "BAD"
        new_mark = "OK " if r["new_top1_correct"] else "BAD"
        out.append(
            f"话题：{r['topic']}　查询：「{r['query']}」　应该想起：「{r['correct']}」\n"
            f"  旧公式[{old_mark}] 选了：「{r['old_top1_pick']}」\n"
            f"  新公式[{new_mark}] 选了：「{r['new_top1_pick']}」"
        )
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
    (RESULTS_DIR / f"{stamp}-memory-recall.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"{stamp}-memory-recall.md").write_text(
        table + "\n\n```\n" + render_rows(rows) + "\n```\n", encoding="utf-8",
    )
    print(f"\n已保存到 evals/results/{stamp}-memory-recall.(json|md)")


if __name__ == "__main__":
    asyncio.run(main())
