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
from dataclasses import dataclass, field
from pathlib import Path

try:  # numpy 是可选的：装了就走向量化的快路径，没装就退回纯 Python，结果一样
    import numpy as _np
except ImportError:  # pragma: no cover - 取决于环境装没装
    _np = None

log = logging.getLogger("townmind.memory")

RECENCY_HALF_LIFE = 120.0  # 秒。演示用的时间尺度；真实游戏里应换成游戏内时间
# 每个 NPC 最多记多少条。这个数以前是 200，卡它的从来不是存储（200 条 JSON 才几百 KB），
# 而是检索：向量化之前一次 recall 要 200ms，10 个 NPC 一轮决策光检索就 2 秒，再往上加
# 就没法玩了。向量化之后同样的检索只要 1.6ms，这个约束就松开了。
#
# 实测决定取值（1536 维真实向量，k=5）：1000 条一次 recall 约 5ms，10 个 NPC 一轮 50ms，
# 相对 LLM 那一下 871ms 的延迟可以忽略。再往上也跑得动（5000 条约 26ms），但记忆本身的
# 质量会先成为瓶颈——实测活跃对话时每分钟产生 26~41 条记忆，其中八成是"你对 X 说了「嗯」"
# 这种逐句对话记录，多存只是多存了一堆流水账。真正该做的是在对话结束时把整场对话压成
# 一条，那之后再考虑要不要继续往上调。
DEFAULT_CAPACITY = 1000
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
# 主动分享：重要度到了这个数的记忆才值得主动跟人提起。定在 6 是因为一般的寒暄、
# "你第一次见到某人"这类日常记录都在这之下，不会让 NPC 见谁都絮叨鸡毛蒜皮的事。
SHARE_MIN_IMPORTANCE = 6
# 传播衰减：每转述一手，重要度打这个折。跟 SHARE_MIN_IMPORTANCE 配合会自然产生一个
# 有意思的性质——消息会自己"传死"：转述几次之后重要度掉到分享门槛以下，就没人再往下
# 传了，而且越是一开始就轰动的消息传得越远。这不是硬写的规则，是两个数凑在一起的结果，
# 具体能传几手由 evals/gossip_propagation.py 实测，不在这里拍脑袋下结论。
HOP_IMPORTANCE_DECAY = 0.8


def retold_importance(source_importance: int) -> int:
    """一条消息被转述一次之后，在听者心里还剩几分重要。

    衰减是"每转述一次打一次折"的递推（i -> round(i * 0.8)），不是按代数一次性指数打折。
    两者看起来等价，其实不是：听者记的是"转述者当时觉得这事有多重要"再打一折，而转述者
    自己那个数也是这么一路折下来的，递推才对得上。按代数指数打折的话，如果听者又同时
    继承了转述者已经打过折的值，就会折两遍。

    至少留 1 分：再怎么传得远，它毕竟还是件"我知道的事"，不该被压成 0 直接消失。"""
    return max(1, round(max(1, min(10, int(source_importance))) * HOP_IMPORTANCE_DECAY))


def _similarity(a: "Memory", b: "Memory") -> float:
    """两条记忆之间的余弦相似度。两边都预先归一化过时就是一次点积；否则退回纯 Python。"""
    if a._vec is not None and b._vec is not None:
        return float(a._vec @ b._vec)
    return _cosine(a.embedding, b.embedding)


def _unit_vector(embedding):
    """把向量归一化成 numpy 行向量；没装 numpy、没有向量、或者是零向量时返回 None。

    归一化放在写入时做是有讲究的：检索时要拿一个查询向量去跟成百上千条记忆比，如果每次
    都现算每条记忆的模长，同一个数会被反复算无数遍。预先归一化之后，余弦相似度就退化成
    一次点积，整批记忆一次矩阵乘就算完了。"""
    if _np is None or embedding is None:
        return None
    v = _np.asarray(embedding, dtype=_np.float64)
    n = float(_np.linalg.norm(v))
    return None if n == 0.0 else v / n


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
    # 传播代数：这件事我是第几手知道的。0 = 亲身经历、或者当事人自己跟我说的；
    # 1 = 听人转述的；2 = 听人转述别人转述的……小道消息传得越远越不可靠，这个数就是
    # "传了多远"的度量。它有两个用处：重要度按代数打折（见 add()），以及提示词里把
    # 二手消息标成"听来的，未必准"（见 agent._build_prompt），让 NPC 转述时自己带上
    # 不确定的口吻，而不是当成亲眼所见言之凿凿地说出去。
    hop: int = 0
    # 这件事我已经跟谁讲过了。NPC 主动跟人搭话、分享自己知道的事时用它去重——
    # 没有这个标记的话，同一件事会被翻来覆去讲给同一个人听，这是"会主动说话的 NPC"
    # 最容易露馅的地方。只记"讲给谁"，不记"讲过几次"：一次就够了。
    told_to: frozenset[str] = frozenset()
    # 这条记忆属于哪条"消息"。玩家私下告诉某个 NPC 一件事时打上一个标签，之后这条消息
    # 传到谁手里，新生成的记忆都继承同一个标签——于是"这条消息现在传到哪儿了、各人嘴里
    # 变成了什么样"就能直接查出来，不用靠文本相似度去猜（那种猜法一转述就失效）。
    # 空字符串 = 不属于任何被追踪的消息，也就是绝大多数日常记忆。
    topic: str = ""
    # 归一化之后的 numpy 行向量，写入时算好，检索时直接做矩阵乘。
    #
    # 为什么不直接把 numpy 数组存进上面的 embedding 字段：dataclass 自动生成的 __eq__ 会
    # 逐字段比较，numpy 数组的 == 返回的是逐元素布尔数组，再被 bool() 一取就抛
    # "truth value of an array is ambiguous"，会炸掉一大片拿 Memory 做相等判断的测试。
    # 所以单开一个字段，并且 compare=False（不参与相等判断）、repr=False（不污染打印）。
    # embedding 仍然是原样的 tuple，落盘、跨进程、没装 numpy 的环境都不受影响。
    _vec: object = field(default=None, compare=False, repr=False)
    # ↑ 这个字段必须排在最后：它是内部缓存，不是记忆本身的属性。踩过一次——一开始把它
    # 插在 told_to 前面，而 load() 是按位置构造 Memory 的，于是读回来的 told_to 被塞进了
    # _vec，记忆"跟谁讲过"的信息全丢了，测试当场抓到。下面 load() 已经改成按名字传参，
    # 以后再加字段不会再踩这个坑，但顺序本身也一并摆正。


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
        normalize_relevance: bool = True,
    ) -> None:
        self.capacity = capacity
        self.half_life = half_life
        self.mmr_lambda = mmr_lambda
        # 相关度是否按本次候选池做 min-max 归一化（见 _ranking_components 的说明）。默认开——
        # 关掉就是加这一步之前的旧行为，留这个开关只为了能做 A/B 对比：同一批记忆、同一个
        # 问题，开和关想起来的东西不一样，这是真实 embedding 上才看得出来的差别。
        self.normalize_relevance = normalize_relevance
        self.memories: list[Memory] = []
        self.met: set[str] = set()  # 已经见过的人，用来判断"第一次见到"
        self.importance_since_reflection: float = 0.0  # 上次反思以来，新记忆的重要度累计到了多少
        self.importance_since_meta_reflection: float = 0.0  # 上次二级反思以来，新增的一级反思累计了多少重要度

    # ---- 存 ----
    def add(
        self, text: str, importance: int, now: float, people=(), embedding=None, kind: str = "event",
        hop: int = 0, topic: str = "",
    ) -> None:
        importance = max(1, min(10, int(importance)))
        # hop 在这里只是"这是第几手"的标记，不参与打分：打折由写入方按 retold_importance()
        # 一次一次折下来（见 agent._remember），这里再折一遍就成了双重折扣
        hop = max(0, int(hop))
        emb = tuple(embedding) if embedding is not None else None
        m = Memory(
            text=text, time=now, importance=importance, people=frozenset(people),
            embedding=emb, kind=kind, hop=hop, topic=topic,
        )
        m._vec = _unit_vector(emb)
        self.memories.append(m)
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
        return [m for m, _ in self._rank_with_components(involved, now, k, query_embedding)]

    def recall_explained(self, involved, now: float, k: int = DEFAULT_TOP_K, query_embedding=None) -> list[dict]:
        """跟 recall() 排序逻辑完全一致，只是把三项子分数也一起带出来——调试和演示专用。
        这里的三项子分数是实际参与排序的版本（见 _ranking_components() 的归一化），不是
        component_scores() 的原始版本；两者在没有语义检索、或候选里带向量的不到两条时
        完全一样，带了归一化才会不同。recall() 本身的返回值（list[Memory]）不变，不影响
        任何现有调用方。"""
        picked = self._rank_with_components(involved, now, k, query_embedding)
        out = []
        for m, (recency, importance, relevance) in picked:
            out.append(
                {
                    "text": m.text,
                    "kind": m.kind,
                    "hop": m.hop,
                    "recency": round(recency, 3),
                    "importance": round(importance, 3),
                    "relevance": round(relevance, 3),
                    "total": round(recency + importance + relevance, 3),
                }
            )
        return out

    def _bulk_relevance(self, query_embedding) -> dict | None:
        """一次矩阵乘算出所有带向量记忆跟本次查询的余弦相似度，返回 {id(记忆): 相似度}。

        没装 numpy、这次查询没带向量、或者候选里带向量的不到两条时返回 None，调用方退回
        原来的逐条计算——两条路径结果一致，只是快慢不同（tests 里有专门的一致性测试）。

        这是检索这条链路上最值钱的一步优化：_cosine() 是纯 Python 的逐元素循环，1536 维
        向量算一次要几千次解释器级别的浮点运算，实测 200 条记忆一次 recall 要 200ms 左右；
        换成一次矩阵乘之后同样的数据只要零点几毫秒。"""
        if _np is None or query_embedding is None:
            return None
        rows = [m for m in self.memories if m._vec is not None]
        if len(rows) < 2:
            return None
        q = _unit_vector(query_embedding)
        if q is None:
            return None
        sims = _np.stack([m._vec for m in rows]) @ q
        return {id(m): float(s) for m, s in zip(rows, sims)}

    def _ranking_components(
        self, involved, now: float, query_embedding=None
    ) -> list[tuple[Memory, tuple[float, float, float]]]:
        """跟 component_scores() 算的是同一件事，多做一步：如果这次查询带了语义向量，把候选里
        "真的算出了语义相关度"（不是退化成认不认人）的那些，按这批候选自己的相关度分布做一次
        min-max 归一化，再参与排序。

        为什么要归一化：新近度、重要度天然就能取到 0~1 的整个量程（重要度 1~10 直接映射，
        新近度随时间自然衰减到 0）。但相关度不是——实测过一次：同一批候选里，明显对题的一条
        相关度是 0.48，两条完全不对题的分别是 0.12、0.14，真实 embedding 对同一种口吻、同一批
        短句算出来的余弦相似度，哪怕内容完全不沾边也有个不低的基线，"最相关"和"完全不相关"之间
        实际拉开的分差，往往比 0~1 这个理论量程窄得多。不归一化的话，相关度会被新近度、重要度
        这两项系统性地压过——哪怕某条记忆明显更对题，"新且重要"的干扰项还是能赢（这正是
        evals/memory_ablation.py 用真实 embedding 跑出来、offline 假向量没测出来的问题）。

        只对这次查询里"真的有 embedding 可比"的候选做归一化（不到两条candidate带向量时不归一化，
        参见下面 recall_explained 的行为，跟没开语义检索一样），退化成认不认人（0 或 1）的候选
        不参与，因为那不是同一种尺度的信号。component_scores()/score() 本身不做这一步，保持成
        一个跟"这次候选池长什么样"无关的、确定性的单条记忆打分（capacity 淘汰用的就是这个，
        淘汰时没有 query_embedding，不受这次改动影响）。"""
        bulk = self._bulk_relevance(query_embedding)
        if bulk is None:
            raw = [[m, list(self.component_scores(m, involved, now, query_embedding))] for m in self.memories]
        else:
            # 相关度已经用一次矩阵乘批量算好了，这里就不能再把 query_embedding 传进
            # component_scores——否则它会逐条再算一遍纯 Python 的余弦，白白付两份钱
            # （第一版就是这么写的，结果只快了 10 倍而不是几百倍）。传 None 进去，让它只算
            # 新近度和重要度；没有向量的那些候选照常退化成"认不认人"，跟原来完全一致。
            raw = []
            for m in self.memories:
                comps = list(self.component_scores(m, involved, now, None))
                s = bulk.get(id(m))
                if s is not None:
                    comps[2] = max(0.0, s)
                raw.append([m, comps])
        if query_embedding is not None and self.normalize_relevance:
            embedded = [i for i, (m, _) in enumerate(raw) if m.embedding is not None]
            if len(embedded) >= 2:
                rels = [raw[i][1][2] for i in embedded]
                lo, hi = min(rels), max(rels)
                if hi - lo > 1e-9:
                    for i in embedded:
                        raw[i][1][2] = (raw[i][1][2] - lo) / (hi - lo)
        return [(m, tuple(comps)) for m, comps in raw]

    def _rank_with_components(
        self, involved, now: float, k: int, query_embedding=None
    ) -> list[tuple[Memory, tuple[float, float, float]]]:
        """recall() / recall_explained() 共用的排序逻辑：先按三项加权总分（用的是
        _ranking_components() 归一化之后的版本）排一次序；如果这次查询带了语义向量、而且
        候选里至少两条记忆有 embedding，再用 MMR 重排一遍 top-k——纯按分数排序不会管"选出来
        的这几条彼此像不像"，好几条内容高度相似的记忆可能会一起挤进来，占掉本该属于"另一件
        事"的名额，MMR 是检索里处理这个问题的经典做法。没配 embedding、或者候选里带向量的
        不到两条，直接退化成原来的纯 top-k，不影响没接语义检索时的行为。"""
        scored = [(m, comps, sum(comps)) for m, comps in self._ranking_components(involved, now, query_embedding)]
        scored.sort(key=lambda triple: triple[2], reverse=True)
        if k <= 0:
            return []
        if query_embedding is None:
            return [(m, comps) for m, comps, _ in scored[:k]]
        embedded_count = sum(1 for m, _, _ in scored if m.embedding is not None)
        if embedded_count < 2:
            return [(m, comps) for m, comps, _ in scored[:k]]
        picked = self._mmr_select([(m, s) for m, _, s in self._mmr_candidates(scored, k)], k)
        comps_by_id = {id(m): comps for m, comps, _ in scored}
        return [(m, comps_by_id[id(m)]) for m in picked]

    def _mmr_candidates(self, scored, k: int):
        """MMR 只需要看"还有可能被选中"的那些候选，剩下的可以直接不参与——结果保证不变。

        这不是"取前 N 名"那种拍脑袋的截断，是从 mmr_lambda 推出来的精确界：

            mmr(m) = λ·base(m) − (1−λ)·penalty(m)，  penalty ∈ [−1, 1]

        所以任何一条候选的 mmr 分数都落在 λ·base(m) ± (1−λ) 这个区间里。低分候选 B 想赢过
        高分候选 A，最好的情况是 B 拿到最低惩罚、A 拿到最高惩罚：

            λ·base(B) + (1−λ) > λ·base(A) − (1−λ)   ⟺   base(A) − base(B) < 2(1−λ)/λ

        也就是说，分数比第 k 名低 2(1−λ)/λ 以上的候选，无论跟已选集合有多不像都赢不了——
        因为每一轮贪心选择时，只要还没选满 k 条，按分数排在前 k 的候选里必定至少有一条还在
        池子里（要么没被选走，要么已经被选走、那它本来就占了一个名额），B 永远输给它。

        这个界会自己适配参数，不需要调：
          λ=1.0  界为 0    → 池子就是前 k 条（MMR 退化成纯 top-k，本来就该这样）
          λ=0.7  界 0.857  → 默认值下池子通常只有几十条，省掉绝大部分计算
          λ=0.2  界 8.0    → 超过总分的量程（三项各 0~1，最多 3），池子退回全量——
                             而这正是"调低 λ 追求多样性"时确实需要全量才算得对的情况

        为什么值得做：_mmr_select 是 O(N·k²) 的，N 是整个记忆库。实测 200 条记忆、1536 维
        真实向量时，一次 recall 要 201ms，其中约 190ms 花在 MMR 上——而排名第 195 的记忆
        根本不可能赢，却每次都要跟已选的逐条算余弦。"""
        if k >= len(scored):
            return scored
        if self.mmr_lambda <= 0.0:
            return scored  # 完全不看分数、只看多样性：谁都可能赢，不能砍
        margin = 2.0 * (1.0 - self.mmr_lambda) / self.mmr_lambda
        cutoff = scored[k - 1][2] - margin
        # scored 已经按总分降序排好，第一个低于 cutoff 的位置之后全都可以不要
        for i in range(k, len(scored)):
            if scored[i][2] < cutoff:
                return scored[:i]
        return scored

    def _mmr_select(self, scored: list[tuple["Memory", float]], k: int) -> list["Memory"]:
        """贪心 MMR：每一步都从剩下的候选里，挑"自身分数高、又跟已经选中的都不太像"的那条。
        mmr_lambda 越接近 1 越只看分数（退化成纯 top-k）；越接近 0 越优先追求多样性。"""
        pool = list(scored)
        selected: list[Memory] = []
        while pool and len(selected) < k:
            best_idx, best_mmr = 0, float("-inf")
            for i, (m, base_score) in enumerate(pool):
                if selected and m.embedding is not None:
                    sims = [_similarity(m, s) for s in selected if s.embedding is not None]
                    penalty = max(sims) if sims else 0.0
                else:
                    penalty = 0.0
                mmr_score = self.mmr_lambda * base_score - (1 - self.mmr_lambda) * penalty
                if mmr_score > best_mmr:
                    best_idx, best_mmr = i, mmr_score
            selected.append(pool.pop(best_idx)[0])
        return selected

    # ---- 主动分享 ----
    def mark_told(self, m: Memory, who: str) -> None:
        """记下"这件事我跟 who 讲过了"，下次就不会再挑中它去跟同一个人讲。"""
        m.told_to = m.told_to | {who}

    def shareable(self, other: str, now: float, min_importance: int = SHARE_MIN_IMPORTANCE, k: int = 1) -> list[Memory]:
        """挑出"值得主动跟 other 说、而且还没跟ta说过"的记忆，按打分从高到低给前 k 条。

        三层筛选，每一层都是为了让主动搭话显得自然而不是机械：
          1. 重要度够高——鸡毛蒜皮的事不值得特意提起；
          2. 还没跟这个人讲过（told_to）——不然同一件事会翻来覆去讲给同一个人听；
          3. 这个人自己不在这条记忆里（other not in people）——不要把对方刚说过的话
             当成新鲜事讲回给ta听，这是最容易让人出戏的一种。

        排序复用 score()（新近度+重要度），不带 query_embedding：这里问的是"我手上有什么
        值得说的事"，不是"跟此刻的话题有多相关"——相关性该由大模型看着当下的对话自己判断，
        检索这一层只负责把够分量的候选捞出来。"""
        pool = [
            m
            for m in self.memories
            if m.importance >= min_importance and other not in m.told_to and other not in m.people
        ]
        pool.sort(key=lambda m: self.score(m, frozenset(), now), reverse=True)
        return pool[: max(0, k)]

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
                    "hop": m.hop,
                    "topic": m.topic,
                    "told_to": sorted(m.told_to),
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
                # 一律按名字传参，不按位置：Memory 的字段是一路加出来的（embedding、kind、
                # hop、told_to…），按位置构造的话，以后任何一次"在中间插一个字段"都会把
                # 后面的值整体错位，而且不会报错，只会让数据静悄悄地跑到错误的字段里去。
                Memory(
                    text=d["text"],
                    time=float(d["time"]),
                    importance=int(d["importance"]),
                    people=frozenset(d["people"]),
                    # 老的记忆文件（加语义检索之前存的）没有这个字段，get 兜底成 None，
                    # 相关度就自动退化回"认不认人"那一版，不会因为读老文件而崩溃
                    embedding=tuple(d["embedding"]) if d.get("embedding") is not None else None,
                    # 老的记忆文件（加分层反思之前存的）没有这个字段，兜底成 "event"——
                    # 不会把老记忆错当成一级反思，二级反思的计数也不会被老数据污染
                    kind=d.get("kind", "event"),
                    # 老的记忆文件（加传播代数之前存的）没有这个字段，兜底成 0——
                    # 当成第一手，跟加这个功能之前的行为一致
                    hop=int(d.get("hop", 0)),
                    # 老的记忆文件（加主动分享之前存的）没有这个字段，兜底成空集合——
                    # 最坏的结果只是这些老记忆有可能被再讲一遍，不会崩
                    told_to=frozenset(d.get("told_to", ())),
                    # 老文件没有这个字段，兜底成空串 = 不属于任何被追踪的消息
                    topic=d.get("topic", ""),
                )
                for d in data["memories"]
            ]
            for m in memories:  # 落盘的是原始 tuple，读回来要重新算一遍归一化向量
                m._vec = _unit_vector(m.embedding)
            store.memories = memories
            store.met = set(data["met"])
            # 老的记忆文件（加反思之前存的）没有这个字段，兜底成 0——大不了这个 NPC
            # 重新攒一轮重要度才触发下一次反思，不会因为读老文件而崩溃
            store.importance_since_reflection = float(data.get("importance_since_reflection", 0.0))
            store.importance_since_meta_reflection = float(data.get("importance_since_meta_reflection", 0.0))
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("memory file %s is unreadable (%s); starting empty", path, e)
        return store
