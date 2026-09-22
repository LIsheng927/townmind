"""记忆检索的消融实验：只用新近度 / 新近度+重要度 / 三项直接相加 / recall() 实际排序
（三项相加 + 跨候选归一化）——同一批记忆、同一个查询，几种配置分别选出的 top-1 差多少。

MemoryStore.score() 是新近度+重要度+相关度三项加权求和，component_scores() 能把三项拆
开看（task 17 加的），这个脚本就是拿这个拆分做消融：每个场景放一条"很久以前的事、重要度
也不高，但内容真的对题"的目标记忆，配两条"很新、重要度也不低，但内容跟问题毫不相关"的
干扰记忆——只看新近度、或者只看新近度+重要度，这两种朴素配置都会被"新且重要"的干扰项
骗到。

这个脚本本来只做到"三项直接相加"这一档就打算收工，offline 假向量下这一档已经是 5/5——
但用真实 embedding 一跑，这一档反而是 0/5！根本原因：offline 用的关键词词袋向量，相关度
不是 0 就是 1，天然跟新近度/重要度同一个量级；真实 embedding 对同一种口吻的短句算出来的
余弦相似度，哪怕内容完全不沾边也有个不低的基线，"最相关"和"完全不相关"之间能拉开的分差
往往只有 0.1~0.5 这么窄，直接相加时新近度+重要度（能到 0~1 满量程）系统性地压过了相关度。
为了解决这个，MemoryStore._ranking_components() 加了一步跨候选的 min-max 归一化（只对
"真的算出语义相关度"的候选生效），这里第四档"recall() 实际排序"就是验证这一步归一化真的
把问题解了——不是自己又搭了一套简化逻辑，走的是 recall() 本身的代码路径。

跟 evals/memory_recall.py 不是一回事：那边比的是"相关度这一项，用旧公式（认不认人）还是
新公式（语义相似度）"，这里比的是"相关度这一项到底值不值得加进总分、加了之后跟只看
新近度/重要度比差多少"——两个脚本问的是不同层次的问题，不重复。

附带测了一下 MMR（task 18 新加的多样性去重，recall() 内部的 _rank()）：把 mmr_lambda 设成
1.0（完全退化成纯相关性排序，等价于没开 MMR）、默认的 0.7、跟调低到 0.4 做三档对比，看
同一个 query 下 top-2 里覆盖了几个不同的话题——纯按分数排序时，两条内容高度相似的记忆会
一起挤进 top-k，把本该属于"另一件事"的名额占掉；默认值偏保守，这个场景里还不足以触发
换人，调低才会。

用法（在 server 目录下）：
  uv run python -m evals.memory_ablation --embedder offline   # 不联网，用内置的关键词词袋向量
  uv run python -m evals.memory_ablation --embedder real       # 配了 OPENAI_API_KEY 就用真实语义向量

跟 evals/memory_recall.py 不一样的地方：这里 offline 模式用的不是纯随机哈希向量，而是一个
基于关键词词袋的确定性向量——每个场景里目标记忆和查询共享至少一个关键词，干扰记忆完全不
共享，offline 模式下就已经能看出有意义的数字，不需要真的配 API key。--embedder real 换成
真实 embedding 之后结论应该更稳，但不是"从能跑变成不能跑"那种区别（memory_recall.py 里
offline 模式是纯随机哈希，数字没有参考意义，这个脚本刻意选了不一样的 offline 实现）。
"""
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from townmind.llm.embeddings import make_embedder
from townmind.memory import MemoryStore

RESULTS_DIR = Path(__file__).parent / "results"
NOW = 10000.0

# 关键词词袋：每个场景内部，目标记忆和查询共享至少一个词，两条干扰记忆完全不共享任何词——
# 场景之间互相独立评测（各自一个全新的 MemoryStore），词会不会在别的场景里重复出现无所谓
VOCAB = [
    "剑", "铁匠", "手艺", "面包", "烘焙", "法棍", "巡逻", "加班", "河边", "钓鱼",
    "天象", "渔村", "下雨", "新剑", "市集", "帮忙", "公所", "账目", "账本", "采摘",
    "浆果", "码头", "看船", "巡夜", "异常", "晚饭", "丰盛", "篱笆", "心情", "哼歌",
    "集市", "摆摊", "卖菜", "涨价",
]

SCENARIOS = [
    {
        "topic": "打剑的手艺",
        "query": "这把剑是谁打的，手艺怎么样",
        "target": "Bob 年轻时候跟着老铁匠学过好几年打剑的手艺",
        "distractors": [
            "Bob 说他今天心情特别好，一直在哼歌",
            "Bob 说他准备去集市摆摊卖菜",
        ],
    },
    {
        "topic": "面包烘焙",
        "query": "这个法棍是怎么烤出来的，好吃吗",
        "target": "Bob 以前在面包店学过烘焙，会做法棍",
        "distractors": [
            "Bob 说他今晚要巡逻，可能会加班",
            "Bob 说他刚去河边钓鱼，钓了不少",
        ],
    },
    {
        "topic": "看天象",
        "query": "是不是快要下雨了，看天象准不准",
        "target": "Bob 说他从小在渔村长大，特别会看天象，一看就知道要下雨",
        "distractors": [
            "Bob 说他刚买了把新剑，很满意",
            "Bob 说他今晚要去市集帮忙",
        ],
    },
    {
        "topic": "账目",
        "query": "账本上这笔钱是怎么回事，账目对不对",
        "target": "Bob 以前在镇公所管过几年账目，账本记得很细",
        "distractors": [
            "Bob 说他刚采摘了一篮子新鲜的浆果",
            "Bob 说他准备去码头看船",
        ],
    },
    {
        "topic": "巡夜",
        "query": "昨晚巡夜的时候有没有发现什么异常",
        "target": "Bob 当过几年镇上的巡夜人，很熟悉夜里的动静，一有异常就能发现",
        "distractors": [
            "Bob 说他刚吃了顿丰盛的晚饭",
            "Bob 说他这几天在忙着修篱笆",
        ],
    },
]

TARGET_IMPORTANCE = 4  # 目标记忆：重要度一般，很久以前的事
TARGET_AGE = 6000.0
DISTRACTOR_IMPORTANCE = 5  # 干扰记忆：重要度不低，而且很新
DISTRACTOR_AGES = (200.0, 350.0)

CONFIGS = {
    "仅新近度": (1, 0, 0),
    "新近度+重要度": (1, 1, 0),
    "三项直接相加（不归一化）": (1, 1, 1),
}
RECALL_LABEL = "recall() 实际排序（三项相加 + 跨候选归一化）"


class OfflineEmbedder:
    """基于关键词词袋的确定性向量，不联网、不花钱。跟 evals/memory_recall.py 的
    OfflineEmbedder（纯随机哈希）不一样：这里特意让目标记忆和查询共享关键词、干扰记忆
    完全不共享，offline 模式下也能看到有意义的差异，不用真的配 API key 才能演示。"""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 if kw in t else 0.0 for kw in VOCAB] for t in texts]


def weighted_score(components: tuple[float, float, float], weights: tuple[int, int, int]) -> float:
    return sum(c * w for c, w in zip(components, weights))


async def eval_scenario(scenario: dict, embedder) -> dict:
    store = MemoryStore()
    store.add(scenario["target"], TARGET_IMPORTANCE, NOW - TARGET_AGE)
    for text, age in zip(scenario["distractors"], DISTRACTOR_AGES):
        store.add(text, DISTRACTOR_IMPORTANCE, NOW - age)

    vectors = await embedder.embed([m.text for m in store.memories] + [scenario["query"]])
    mem_vecs, query_vec = vectors[:-1], vectors[-1]
    for m, v in zip(store.memories, mem_vecs):
        m.embedding = tuple(v)

    result = {"topic": scenario["topic"], "query": scenario["query"], "correct": scenario["target"]}
    for label, weights in CONFIGS.items():
        ranked = sorted(
            store.memories,
            key=lambda m: weighted_score(store.component_scores(m, frozenset(), NOW, query_vec), weights),
            reverse=True,
        )
        top1 = ranked[0].text
        result[label] = {"top1_pick": top1, "correct": top1 == scenario["target"]}

    # 第四档：不自己手算加权，直接走 recall() 本身（含 _ranking_components() 的归一化）
    top1 = store.recall(frozenset(), NOW, k=1, query_embedding=query_vec)[0].text
    result[RECALL_LABEL] = {"top1_pick": top1, "correct": top1 == scenario["target"]}
    return result


async def eval_mmr_diversity(embedder) -> dict:
    """MMR 多样性去重的效果：3 条记忆里有 2 条内容高度相似（都在说"集市要涨价"，跟查询的
    相关度都是满分、彼此之间也高度相似)，1 条是有点沾边但不完全一样的话题（集市旁边的广场
    在维修，相关度没那么高，但跟那两条重复记忆也不是一回事）。

    只看 top-2：纯按相关性排序（mmr_lambda=1.0，等价于没开 MMR）会把两条几乎重复的记忆都
    选进来——同一件事说了两遍，等于只带来 1 条增量信息。这里额外把默认值（mmr_lambda=0.7）
    也摆出来对比，而不是只摆"关掉 vs 调到很低"两组——0.7 是刻意选的保守默认值（以相关性
    为主），这个场景里两条重复记忆和"广场维修"那条的相关度差距还不小，默认值不足以把
    "广场维修"换进 top-2，调低到 0.4 才会换——这个三档对比本身就是"mmr_lambda 这个超参数
    在做什么"最直观的演示，比只给一组"开/关"对比更说明问题。
    """
    store = MemoryStore()
    texts = [
        "集市上好多摊位都在说下个月要涨价",
        "听说集市摊位下个月起要涨价了",  # 跟上一条几乎是同一件事，相关度也跟上一条一样高
        "集市旁边的广场最近在维修地砖，绕行了好几天",  # 沾一点边（提到集市），但话题不一样
    ]
    for t in texts:
        store.add(t, importance=6, now=NOW)
    query = "集市最近是不是要涨价了"
    vectors = await embedder.embed(texts + [query])
    mem_vecs, query_vec = vectors[:-1], vectors[-1]
    for m, v in zip(store.memories, mem_vecs):
        m.embedding = tuple(v)

    def topics_covered(picked: list) -> int:
        # 粗略地把"涨价"算一个话题、"维修"算另一个、"钓鱼"算第三个，数 top-2 里覆盖了几个
        buckets = [{"涨价"}, {"维修"}, {"钓鱼"}]
        covered = set()
        for m in picked:
            for i, bucket in enumerate(buckets):
                if any(kw in m.text for kw in bucket):
                    covered.add(i)
        return len(covered)

    variants = {
        "关掉（mmr_lambda=1.0，等价于纯相关性排序）": 1.0,
        "默认（mmr_lambda=0.7，保守，偏向相关性）": 0.7,
        "调低多样性权重（mmr_lambda=0.4）": 0.4,
        "调得更低（mmr_lambda=0.2）": 0.2,
        "几乎只看多样性（mmr_lambda=0.1）": 0.1,
    }
    out = {}
    for label, lam in variants.items():
        s = MemoryStore(mmr_lambda=lam)
        s.memories = store.memories
        picked = s.recall(frozenset(), NOW, k=2, query_embedding=query_vec)
        out[label] = {"picks": [m.text for m in picked], "topics_covered": topics_covered(picked)}
    return out


async def run(embedder_kind: str) -> tuple[dict, list[dict], dict]:
    if embedder_kind == "offline":
        embedder = OfflineEmbedder()
    else:
        embedder = make_embedder()
        if embedder is None:
            raise SystemExit("没有可用的 embedding 客户端：请检查 server/.env 里是否配置了 OPENAI_API_KEY。")
    rows = [await eval_scenario(s, embedder) for s in SCENARIOS]
    n = len(rows)
    all_labels = [*CONFIGS, RECALL_LABEL]
    summary = {
        label: sum(1 for r in rows if r[label]["correct"]) / n
        for label in all_labels
    }
    mmr = await eval_mmr_diversity(embedder)
    return summary, rows, mmr


def render_table(summary: dict, n: int) -> str:
    lines = ["| 配置 | Top-1 命中目标记忆 |", "|---|---|"]
    for label, acc in summary.items():
        lines.append(f"| {label} | {acc:.0%}（{round(acc * n)}/{n}） |")
    return "\n".join(lines)


def render_rows(rows: list[dict]) -> str:
    out = []
    for r in rows:
        out.append(f"话题：{r['topic']}　查询：「{r['query']}」　应该想起：「{r['correct']}」")
        for label in [*CONFIGS, RECALL_LABEL]:
            mark = "OK " if r[label]["correct"] else "BAD"
            out.append(f"  [{mark}] {label}：选了「{r[label]['top1_pick']}」")
    return "\n".join(out)


def render_mmr(mmr: dict) -> str:
    blocks = []
    for label, data in mmr.items():
        picks = "\n".join(f"  - {t}" for t in data["picks"])
        blocks.append(f"{label} top-{len(data['picks'])}：\n{picks}\n  覆盖话题数：{data['topics_covered']} / {len(data['picks'])}")
    return "\n\n".join(blocks)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedder", choices=["offline", "real"], default="offline")
    args = ap.parse_args()
    summary, rows, mmr = await run(args.embedder)
    table = render_table(summary, len(rows))
    print(f"\nembedder={args.embedder}\n\n{table}\n\n{render_rows(rows)}\n\n--- MMR 多样性去重 ---\n\n{render_mmr(mmr)}")
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}-memory-ablation.json").write_text(
        json.dumps({"args": vars(args), "summary": summary, "rows": rows, "mmr": mmr}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"{stamp}-memory-ablation.md").write_text(
        table + "\n\n```\n" + render_rows(rows) + "\n\n" + render_mmr(mmr) + "\n```\n", encoding="utf-8",
    )
    print(f"\n已保存到 evals/results/{stamp}-memory-ablation.(json|md)")


if __name__ == "__main__":
    asyncio.run(main())
