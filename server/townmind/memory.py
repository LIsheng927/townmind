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
    def __init__(self, capacity: int = DEFAULT_CAPACITY, half_life: float = RECENCY_HALF_LIFE) -> None:
        self.capacity = capacity
        self.half_life = half_life
        self.memories: list[Memory] = []
        self.met: set[str] = set()  # 已经见过的人，用来判断"第一次见到"

    # ---- 存 ----
    def add(self, text: str, importance: int, now: float, people=(), embedding=None) -> None:
        importance = max(1, min(10, int(importance)))
        emb = tuple(embedding) if embedding is not None else None
        self.memories.append(Memory(text, now, importance, frozenset(people), emb))
        if len(self.memories) > self.capacity:
            # 容量满了：淘汰"又旧又不重要"的那条（不看相关度，因为它此刻与谁有关/跟什么话题有关不重要）
            worst = min(range(len(self.memories)), key=lambda i: self.score(self.memories[i], frozenset(), now))
            self.memories.pop(worst)

    # ---- 取 ----
    def score(self, m: Memory, involved, now: float, query_embedding=None) -> float:
        recency = 0.5 ** (max(0.0, now - m.time) / self.half_life)
        importance = m.importance / 10
        if query_embedding is not None and m.embedding is not None:
            # 语义相关度：跟"此刻在聊什么"算余弦相似度，负的（不相关到反着来）按 0 算，
            # 跟旧版"0 或 1"的量级对齐，不会让这一项一家独大盖过新近度和重要度
            relevance = max(0.0, _cosine(query_embedding, m.embedding))
        else:
            relevance = 1.0 if (m.people & frozenset(involved)) else 0.0
        return recency + importance + relevance

    def recall(self, involved, now: float, k: int = DEFAULT_TOP_K, query_embedding=None) -> list[Memory]:
        ranked = sorted(self.memories, key=lambda m: self.score(m, involved, now, query_embedding), reverse=True)
        return ranked[:k]

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
            "memories": [
                {
                    "text": m.text,
                    "time": m.time,
                    "importance": m.importance,
                    "people": sorted(m.people),
                    "embedding": list(m.embedding) if m.embedding is not None else None,
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
                )
                for d in data["memories"]
            ]
            store.memories = memories
            store.met = set(data["met"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("memory file %s is unreadable (%s); starting empty", path, e)
        return store
