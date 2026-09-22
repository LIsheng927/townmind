"""NPC 的长期记忆：存、取、淘汰、落盘。

每条记忆有：内容、发生时间、重要度（1-10）、涉及的人、（可选的）语义向量。
每次做决策前，给所有记忆打分，只取分数最高的几条放进提示词。总分由三项相加（这是
Stanford「Generative Agents」论文里记忆检索用的同一个公式）：
  新近度（0~1）：越新越高，每过 half_life 秒减半
  重要度（0~1）：重要度 / 10
  相关度（0~1）：这条记忆的内容，跟"此刻在聊什么"语义上有多像——
    配了 embedding 客户端（townmind/llm/embeddings.py）时，用当前情境的向量和这条记忆的
    向量算余弦相似度；没配（没设 OPENAI_API_KEY）时，退化成旧的粗糙版本：
    这条记忆涉及的人，此刻是否在附近或刚跟我说话（命中记 1，没命中记 0）——不影响服务启动，
    只是相关度这一项从"懂语义"退化成"认不认人"。
"""
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("townmind.memory")

RECENCY_HALF_LIFE = 120.0  # 秒。演示用的时间尺度；真实游戏里应换成游戏内时间
DEFAULT_CAPACITY = 200
DEFAULT_TOP_K = 5
# 反思（同样出自 Stanford 那篇论文）：累计重要度一旦过了这个数，就该停下来回顾一遍最近的事、
# 提炼出更高层次的认识了——数值跟论文里"重要度分数之和越过阈值"的量级保持一致。
REFLECTION_THRESHOLD = 150.0
# 分层反思（同样出自那篇论文里的 reflection tree）：普通反思（下面这个阈值）是"从具体记忆
# 里提炼感想"；这里是"从好几条感想里再提炼一层更抽象的认识"——累计的是"反思"这个 kind
# 本身的重要度，数值比 REFLECTION_THRESHOLD 更高，因为攒够足够多条一级感想才值得再往上提炼
# 一层，不然每次反思完马上又反思一次意义不大。
META_REFLECTION_THRESHOLD = 300.0
# MMR（Maximal Marginal Relevance）：top-k 排序完之后，会不会因为好几条记忆内容高度相似
# 而一起挤进来、占掉本该属于"另一件事"的名额。lambda 越接近 1，越只看分数本身（退化成
# 纯 top-k）；越接近 0，越优先追求多样性。0.7 是个"以相关性为主、但不完全无视重复"的取值。
MMR_LAMBDA = 0.7


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """两个向量的余弦相似度，取值理论上在 [-1, 1]；某条向量全是 0（理论上不该发生）时算 0 分，不报错。"""
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Memory:
    text: str
    time: float
    importance: int  # 1-10
    people: frozenset[str] = frozenset()
    embedding: tuple[float, ...] | None = None  # 语义向量；没配 embedding 客户端时恒为 None
    # "event"：普通记忆（发生的事、说过的话）；"reflection"：一级反思（从若干 event 里提炼出的
    # 感想）；"meta_reflection"：二级反思（从若干 reflection 里再提炼出的更抽象的认识）。
    # 只用来决定"这条记忆该不该被算进分层反思的选材/计数里"，不影响 recall() 的排序——三种
    # kind 在检索时一视同仁，都是靠新近度+重要度+相关度三项打分。
    kind: str = "event"


def format_age(seconds: float) -> str:
    """把"多久以前"转成大模型和人都容易读的说法。"""
    if seconds < 10:
        return "刚才"
    if seconds < 60:
        return f"{int(seconds)} 秒前"
    if seconds < 3600:
        return f"{int(seconds // 60)} 分钟前"
    return f"{int(seconds // 3600)} 小时前"


class MemoryStore:
    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        half_life: float = RECENCY_HALF_LIFE,
        mmr_lambda: float = MMR_LAMBDA,
    ) -> None:
        self.capacity = capacity
        self.half_life = half_life
        self.mmr_lambda = mmr_lambda
        self.memories: list[Memory] = []
        self.met: set[str] = set()  # 已经见过的人，用来判断"第一次见到"
        self.importance_since_reflection: float = 0.0  # 上次反思以来，新记忆的重要度累计到了多少
        self.importance_since_meta_reflection: float = 0.0  # 上次二级反思以来，新增的一级反思累计了多少重要度

    # ---- 存 ----
    def add(self, text: str, importance: int, now: float, people=(), embedding=None, kind: str = "event") -> None:
        importance = max(1, min(10, int(importance)))
        emb = tuple(embedding) if embedding is not None else None
        self.memories.append(Memory(text, now, importance, frozenset(people), emb, kind))
        self.importance_since_reflection += importance
        if kind == "reflection":
            # 只有"一级反思"才计入二级反思的累计——普通事件、和二级反思本身都不算，
            # 不然会变成"随便攒点日常小事就又反思一次"，失去分层的意义
            self.importance_since_meta_reflection += importance
        if len(self.memories) > self.capacity:
            # 容量满了：淘汰"又旧又不重要"的那条（不看相关度，因为它此刻与谁有关/跟什么话题有关不重要）
            worst = min(range(len(self.memories)), key=lambda i: self.score(self.memories[i], frozenset(), now))
            self.memories.pop(worst)

    # ---- 反思 ----
    def should_reflect(self, threshold: float = REFLECTION_THRESHOLD) -> bool:
        return self.importance_since_reflection >= threshold

    def mark_reflected(self) -> None:
        self.importance_since_reflection = 0.0

    def should_meta_reflect(self, threshold: float = META_REFLECTION_THRESHOLD) -> bool:
        return self.importance_since_meta_reflection >= threshold

    def mark_meta_reflected(self) -> None:
        self.importance_since_meta_reflection = 0.0

    # ---- 取 ----
    def component_scores(self, m: Memory, involved, now: float, query_embedding=None) -> tuple[float, float, float]:
        """新近度、重要度、相关度三项分开返回（score() 就是把这三个加起来）。拆开单独暴露
        出来是为了能回答"这条记忆凭什么被想起来"——调试、给人演示、或者做消融实验
        （对比只用哪几项排序）都要用到这个拆分，而不是只有一个糊在一起的总分。"""
        recency = 0.5 ** (max(0.0, now - m.time) / self.half_life)
        importance = m.importance / 10
        if query_embedding is not None and m.embedding is not None:
            # 语义相关度：跟"此刻在聊什么"算余弦相似度，负的（不相关到反着来）按 0 算，
            # 跟旧版"0 或 1"的量级对齐，不会让这一项一家独大盖过新近度和重要度
            relevance = max(0.0, _cosine(query_embedding, m.embedding))
        else:
            relevance = 1.0 if (m.people & frozenset(involved)) else 0.0
        return recency, importance, relevance

    def score(self, m: Memory, involved, now: float, query_embedding=None) -> float:
        return sum(self.component_scores(m, involved, now, query_embedding))

    def recall(self, involved, now: float, k: int = DEFAULT_TOP_K, query_embedding=None) -> list[Memory]:
        return self._rank(involved, now, k, query_embedding)

    def recall_explained(self, involved, now: float, k: int = DEFAULT_TOP_K, query_embedding=None) -> list[dict]:
        """跟 recall() 排序逻辑完全一致，只是把三项子分数也一起带出来——调试和演示专用。
        recall() 本身的返回值（list[Memory]）不变，不影响任何现有调用方。"""
        picked = self._rank(involved, now, k, query_embedding)
        out = []
        for m in picked:
            recency, importance, relevance = self.component_scores(m, involved, now, query_embedding)
            out.append(
                {
                    "text": m.text,
                    "recency": round(recency, 3),
                    "importance": round(importance, 3),
                    "relevance": round(relevance, 3),
                    "total": round(recency + importance + relevance, 3),
                }
            )
        return out

    def _rank(self, involved, now: float, k: int, query_embedding=None) -> list[Memory]:
        """recall() / recall_explained() 共用的排序逻辑：先按三项加权总分排一次序；如果这次
        查询带了语义向量、而且候选里至少两条记忆有 embedding，再用 MMR 重排一遍 top-k——纯按
        分数排序不会管"选出来的这几条彼此像不像"，好几条内容高度相似的记忆可能会一起挤进来，
        占掉本该属于"另一件事"的名额，MMR 是检索里处理这个问题的经典做法。没配 embedding、
        或者候选里带向量的不到两条，直接退化成原来的纯 top-k，不影响没接语义检索时的行为。"""
        scored = [(m, self.score(m, involved, now, query_embedding)) for m in self.memories]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        if k <= 0:
            return []
        if query_embedding is None:
            return [m for m, _ in scored[:k]]
        embedded_count = sum(1 for m, _ in scored if m.embedding is not None)
        if embedded_count < 2:
            return [m for m, _ in scored[:k]]
        return self._mmr_select(scored, k)

    def _mmr_select(self, scored: list[tuple["Memory", float]], k: int) -> list["Memory"]:
        """贪心 MMR：每一步都从剩下的候选里，挑"自身分数高、又跟已经选中的都不太像"的那条。
        mmr_lambda 越接近 1 越只看分数（退化成纯 top-k）；越接近 0 越优先追求多样性。"""
        pool = list(scored)
        selected: list[Memory] = []
        while pool and len(selected) < k:
            best_idx, best_mmr = 0, float("-inf")
            for i, (m, base_score) in enumerate(pool):
                if selected and m.embedding is not None:
                    sims = [_cosine(m.embedding, s.embedding) for s in selected if s.embedding is not None]
                    penalty = max(sims) if sims else 0.0
                else:
                    penalty = 0.0
                mmr_score = self.mmr_lambda * base_score - (1 - self.mmr_lambda) * penalty
                if mmr_score > best_mmr:
                    best_idx, best_mmr = i, mmr_score
            selected.append(pool.pop(best_idx)[0])
        return selected

    # ---- 调试 ----
    def dump(self, now: float, limit: int = 50) -> list[dict]:
        newest = sorted(self.memories, key=lambda m: m.time, reverse=True)[:limit]
        return [
            {
                "text": m.text,
                "age_seconds": round(now - m.time, 1),
                "importance": m.importance,
                "people": sorted(m.people),
            }
            for m in newest
        ]

    # ---- 落盘 ----
    def save(self, path: Path) -> None:
        data = {
            "met": sorted(self.met),
            "importance_since_reflection": self.importance_since_reflection,
            "importance_since_meta_reflection": self.importance_since_meta_reflection,
            "memories": [
                {
                    "text": m.text,
                    "time": m.time,
                    "importance": m.importance,
                    "people": sorted(m.people),
                    "embedding": list(m.embedding) if m.embedding is not None else None,
                    "kind": m.kind,
                }
                for m in self.memories
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # 先写临时文件再替换，写到一半崩溃也不会损坏原文件

    @classmethod
    def load(cls, path: Path, **kwargs) -> "MemoryStore":
        store = cls(**kwargs)
        if not path.exists():
            return store
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            memories = [
                Memory(
                    d["text"],
                    float(d["time"]),
                    int(d["importance"]),
                    frozenset(d["people"]),
                    # 老的记忆文件（加语义检索之前存的）没有这个字段，get 兜底成 None，
                    # 相关度就自动退化回"认不认人"那一版，不会因为读老文件而崩溃
                    tuple(d["embedding"]) if d.get("embedding") is not None else None,
                    # 老的记忆文件（加分层反思之前存的）没有这个字段，兜底成 "event"——
                    # 不会把老记忆错当成一级反思，二级反思的计数也不会被老数据污染
                    d.get("kind", "event"),
                )
                for d in data["memories"]
            ]
            store.memories = memories
            store.met = set(data["met"])
            # 老的记忆文件（加反思之前存的）没有这个字段，兜底成 0——大不了这个 NPC
            # 重新攒一轮重要度才触发下一次反思，不会因为读老文件而崩溃
            store.importance_since_reflection = float(data.get("importance_since_reflection", 0.0))
            store.importance_since_meta_reflection = float(data.get("importance_since_meta_reflection", 0.0))
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("memory file %s is unreadable (%s); starting empty", path, e)
        return store
