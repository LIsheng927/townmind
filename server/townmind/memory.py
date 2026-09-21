"""NPC 的长期记忆：存、取、淘汰、落盘。

每条记忆有：内容、发生时间、重要度（1-10）、涉及的人。
每次做决策前，给所有记忆打分，只取分数最高的几条放进提示词。总分由三项相加：
  新近度（0~1）：越新越高，每过 half_life 秒减半
  重要度（0~1）：重要度 / 10
  相关度（0 或 1）：这条记忆涉及的人，此刻是否在附近或刚跟我说话
"""
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("townmind.memory")

RECENCY_HALF_LIFE = 120.0  # 秒。演示用的时间尺度；真实游戏里应换成游戏内时间
DEFAULT_CAPACITY = 200
DEFAULT_TOP_K = 5


@dataclass
class Memory:
    text: str
    time: float
    importance: int  # 1-10
    people: frozenset[str] = frozenset()


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
    def add(self, text: str, importance: int, now: float, people=()) -> None:
        importance = max(1, min(10, int(importance)))
        self.memories.append(Memory(text, now, importance, frozenset(people)))
        if len(self.memories) > self.capacity:
            # 容量满了：淘汰"又旧又不重要"的那条（不看相关度，因为它此刻与谁有关不重要）
            worst = min(range(len(self.memories)), key=lambda i: self.score(self.memories[i], frozenset(), now))
            self.memories.pop(worst)

    # ---- 取 ----
    def score(self, m: Memory, involved, now: float) -> float:
        recency = 0.5 ** (max(0.0, now - m.time) / self.half_life)
        importance = m.importance / 10
        relevance = 1.0 if (m.people & frozenset(involved)) else 0.0
        return recency + importance + relevance

    def recall(self, involved, now: float, k: int = DEFAULT_TOP_K) -> list[Memory]:
        ranked = sorted(self.memories, key=lambda m: self.score(m, involved, now), reverse=True)
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
                {"text": m.text, "time": m.time, "importance": m.importance, "people": sorted(m.people)}
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
                Memory(d["text"], float(d["time"]), int(d["importance"]), frozenset(d["people"]))
                for d in data["memories"]
            ]
            store.memories = memories
            store.met = set(data["met"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("memory file %s is unreadable (%s); starting empty", path, e)
        return store
