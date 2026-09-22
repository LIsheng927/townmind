"""NPC 之间的关系：好感度和信任度。

记忆解决的是"我记得什么"，这个模块解决的是"我对着谁在说话"——同一件事，跟熟人说和跟
刚认识的人说，愿不愿意说、用什么语气说都该不一样。小镇要显得"活"，靠的正是这一层：
每个 NPC 对每个人维护一份独立的印象，而不是对所有人一视同仁。

两个维度故意分开，不合成一个"亲密度"：
  好感（affinity）：喜不喜欢这个人——影响语气热不热络。
  信任（trust）：信不信这个人说的话——影响愿不愿意把自己知道的事告诉他，
    也影响听到他转述的消息时打几折（见 agent.py 里的八卦传播）。
喜欢和信任并不是一回事：一个人可以很讨喜但满嘴跑火车，也可以很闷但说话算数。分开之后，
"愿意跟他闲聊"和"愿意把正事告诉他"就成了两个可以分别触发的行为。

两条机制值得单独说明，它们是这个模块里仅有的"算法"：

1. 有界更新（越接近边界越难推动）。每次对话结束后由大模型给一个 delta，如果直接相加，
   几十次对话之后所有关系都会打满边界、再也区分不出亲疏，而且一两次极端对话就能把关系
   永久钉死。这里改成按"这个方向还剩多少空间"缩放：delta 往当前值已经很极端的方向推时
   几乎推不动，往回拉时不受限制。结果是关系值会收敛在一个区间里，而不是单调漂移到边界。

2. 随时间回归中性。很久不见的人，印象会慢慢淡回 0（半衰期 RELATION_HALF_LIFE）。
   不加这个的话，"三个月前吵过一架"和"刚刚吵过一架"对当下行为的影响一模一样，那就不像
   人了；加了之后，"最近有没有互动"本身才有意义。衰减是读的时候按时间算出来的，不靠
   定时任务去刷新每一对关系——NPC 多起来之后那种刷新是纯粹的浪费。
"""
import logging
from dataclasses import dataclass

log = logging.getLogger("townmind.social")

RELATION_BOUND = 10.0  # 好感/信任的取值范围是 [-RELATION_BOUND, RELATION_BOUND]
RELATION_HALF_LIFE = 1800.0  # 秒。没有互动的话，每过这么久，印象往中性回一半
MAX_NOTES = 3  # 每段关系最多留几条"为什么会这样"的理由，给提示词用；只留最近的


@dataclass
class Relationship:
    """我对某一个人的印象。所有数值都是"截至 last_update 那一刻"的，读的时候要按时间衰减，
    不要直接拿 affinity/trust 字段用——用 RelationshipBook.get() 拿衰减之后的版本。"""

    affinity: float = 0.0
    trust: float = 0.0
    interactions: int = 0  # 一共更新过几次，用来区分"还不熟"和"处久了刚好平淡"
    last_update: float = 0.0
    notes: tuple[str, ...] = ()  # 最近几次变化的理由（大模型给的），最新的在最后


def _decay(value: float, elapsed: float, half_life: float) -> float:
    """把印象往中性（0）拉。elapsed 为负（时钟回拨之类）时按 0 处理，不会反向放大。"""
    if half_life <= 0:
        return value
    return value * 0.5 ** (max(0.0, elapsed) / half_life)


def _saturating_add(old: float, delta: float, bound: float) -> float:
    """有界更新：delta 往"当前值已经很极端"的方向推时，越接近边界越推不动；往回拉时不打折。

    比如 bound=10、old=9（已经很喜欢了）时，再来一次 +3 只会加 0.3，涨到 9.3；但一次 -3
    会实打实减 3，掉到 6——这符合直觉：好感已经很高时再多夸一句没什么增量，但一次失望
    是会实实在在掉分的。反过来也一样。"""
    if bound <= 0:
        return 0.0
    if delta > 0:
        room = max(0.0, 1.0 - old / bound)
    elif delta < 0:
        room = max(0.0, 1.0 + old / bound)
    else:
        return old
    # room 只在 delta 把值往边界推时才 <1；往回拉时 room 会 >1，这里夹到 1，
    # 避免"从 -10 往回拉"被放大成两倍
    return max(-bound, min(bound, old + delta * min(1.0, room)))


def _level(value: float, words: tuple[str, str, str, str, str]) -> str:
    """把 [-bound, bound] 的数映射成一个词。分界点取得比较宽，避免数值一点点波动就换说法，
    让提示词里的描述在几轮对话之间是稳定的。"""
    if value <= -5.0:
        return words[0]
    if value <= -1.5:
        return words[1]
    if value < 1.5:
        return words[2]
    if value < 5.0:
        return words[3]
    return words[4]


AFFINITY_WORDS = ("很反感", "有点不待见", "谈不上喜欢也谈不上讨厌", "还挺有好感", "很喜欢")
TRUST_WORDS = ("完全不信", "不太信得过", "说不上信不信", "比较信得过", "很信任")


class RelationshipBook:
    """一个 NPC 对所有人的印象。每个 NPC 一本，跟 MemoryStore 一样按 npc_id 分开存。"""

    def __init__(self, half_life: float = RELATION_HALF_LIFE, bound: float = RELATION_BOUND) -> None:
        self.half_life = half_life
        self.bound = bound
        self._rels: dict[str, Relationship] = {}

    def known(self) -> list[str]:
        return sorted(self._rels)

    def get(self, other: str, now: float) -> Relationship:
        """拿到"此刻"的印象——已经按距离上次更新的时间做过衰减。没打过交道的人返回一份
        全 0 的空关系（不写进字典，避免只是查一下就凭空多出一堆空关系）。"""
        rel = self._rels.get(other)
        if rel is None:
            return Relationship()
        elapsed = now - rel.last_update
        return Relationship(
            affinity=_decay(rel.affinity, elapsed, self.half_life),
            trust=_decay(rel.trust, elapsed, self.half_life),
            interactions=rel.interactions,
            last_update=rel.last_update,
            notes=rel.notes,
        )

    def apply(self, other: str, affinity_delta: float, trust_delta: float, reason: str, now: float) -> Relationship:
        """一次互动之后更新印象：先按时间衰减到此刻，再做有界更新，最后把时间戳推到此刻。

        顺序很重要——先衰减再加 delta，这样"很久没见、今天聊得不错"的结果是从淡下来的
        基线往上走，而不是从三个月前那个旧值往上走。"""
        current = self.get(other, now)
        rel = Relationship(
            affinity=_saturating_add(current.affinity, float(affinity_delta), self.bound),
            trust=_saturating_add(current.trust, float(trust_delta), self.bound),
            interactions=current.interactions + 1,
            last_update=now,
            notes=(current.notes + (reason,))[-MAX_NOTES:] if reason else current.notes,
        )
        self._rels[other] = rel
        return rel

    def trust_of(self, other: str, now: float) -> float:
        return self.get(other, now).trust

    def describe(self, other: str, now: float, name: str | None = None) -> str | None:
        """给提示词用的一句话。完全没打过交道时返回 None——与其写"你对他没有印象"占一行
        提示词，不如干脆不写。"""
        rel = self.get(other, now)
        if rel.interactions == 0:
            return None
        who = name or other
        line = f"你对{who}的印象：{_level(rel.affinity, AFFINITY_WORDS)}，{_level(rel.trust, TRUST_WORDS)}"
        if rel.notes:
            line += f"（最近一次：{rel.notes[-1]}）"
        return line + "。"

    # ---- 落盘 ----
    def to_dict(self) -> dict:
        return {
            "relationships": {
                other: {
                    "affinity": rel.affinity,
                    "trust": rel.trust,
                    "interactions": rel.interactions,
                    "last_update": rel.last_update,
                    "notes": list(rel.notes),
                }
                for other, rel in self._rels.items()
            }
        }

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "RelationshipBook":
        book = cls(**kwargs)
        if not isinstance(data, dict):
            return book
        for other, d in (data.get("relationships") or {}).items():
            try:
                book._rels[other] = Relationship(
                    affinity=float(d["affinity"]),
                    trust=float(d["trust"]),
                    interactions=int(d.get("interactions", 0)),
                    last_update=float(d.get("last_update", 0.0)),
                    notes=tuple(d.get("notes") or ()),
                )
            except (KeyError, TypeError, ValueError) as e:
                # 单条读坏了就跳过这一条，不要让整本关系册跟着丢——跟 MemoryStore.load
                # 的取舍一致：关系数据是锦上添花，坏了不该影响 NPC 还能不能正常说话
                log.warning("关系数据里 %s 这条读不出来（%s: %s），跳过", other, type(e).__name__, e)
        return book
