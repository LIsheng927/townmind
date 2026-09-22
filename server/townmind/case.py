"""案件模式 v2——"罗生门"式证词调查：不是每个 NPC 各自藏一条预设秘密等玩家解锁，
而是 5 个人对案发当晚分别有一份不完整、甚至有偏差的记忆，玩家要在这些证词之间找
矛盾、找印证，自己拼出真相——这才是"长期记忆系统"和"NPC 不该轻信自己的记忆"这两样
在小镇 Agent 上投入最大的技术，真正被案件模式用上，而不是被晾在一边没用上。

跟 v1（这个模块早期版本，已经被这版取代）最大的区别：
  v1：每个 NPC 一条 Secret（真相 truth / 谎言 cover_story），坦白条件一成立，直接从
      "谎言"跳到"真相"，NPC 自己永远知道确凿的答案，只是要不要说——本质还是一棵
      预设好的对话树，记忆系统、guard 那套东西完全用不上。
  v2：每条 Recollection 有独立的 fact_id + accuracy（witnessed 亲历且准确 / distorted
      亲历但自己理解有偏差，深信不疑 / secondhand 听来的，转述里可能走样）+ 是否
      withheld（知道但故意不说，这一项跟 accuracy 是两回事，正交组合）。distorted/
      secondhand 的记忆不是"谎言"——NPC 自己确信无疑地说出来，只有玩家拿别处印证到
      的矛盾信息去对质，才会让 ta"意识到自己可能记错/听错了"，转向不确定的口吻——
      但不会凭空说出正确答案（正确答案得去问真正 witnessed 那个人）。这条转变路径
      直接复刻 townmind/agent.py 里 distrust_own_memory + _learn_from_correction 的
      思路：不是空泛地提醒"你的记忆不可靠"，而是针对这一条具体内容给出具体的纠正信号。

每个 NPC 身上除了案情相关的 Recollection，也塞了一两条无关痛痒的背景记忆（比如"上周
镇长来借了本书"），跟 Recollection 一起进同一个真正的 townmind.memory.MemoryStore——
让检索机制（新近度 + 重要度 + 相关度那套公式，配了 OPENAI_API_KEY 时还真的走语义检索）
在这个案件里有实打实的活干：问到案情相关的话题，重要度更高的 Recollection 会被优先
想起来；问到不相关的事，NPC 也能聊点别的，不会看起来像"除了这一条秘密什么都不知道"
的工具人。玩家的问话、NPC 自己的回答，也都作为普通记忆存进同一个 store，对话历史本身
就是"记忆"的一部分，不是额外维护的一份聊天记录。

没有接进来的：guard_model.py 里那个专门在小镇场景上训练出来的 LoRA 分类器——它是照着
小镇的人设/编造模式训的，直接套到案件这几个全新角色身上没有意义，要真的用得靠重新
标数据训练，这版先不做，先看这套"结构化记忆状态机 + LLM 现场判定转变条件"够不够用。
"""
import logging
import time as _time
from collections import defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from . import safety
from .llm.base import LLMClient
from .memory import MemoryStore, format_age

log = logging.getLogger("townmind.case")

MAX_REPLY_CHARS = 80  # 案件里 NPC 交代缘由有时候比小镇闲聊台词长，给宽松些的硬上限（提示词里仍要求不超过 40 字）
BACKSTORY_AGE = 3 * 24 * 3600.0  # 案发到问话隔了大概三天，案情相关记忆按这个"年龄"存进去，问话过程中新产生的
# 对话记忆才会明显更"新"，新近度这一项才有意义——不然所有记忆时间戳都差不多，这个维度就是摆设
CASE_IMPORTANCE = 8  # 案情相关记忆的重要度，跟 agent.py 的 IMPORTANCE_MET 同一量级——够重要，会被优先想起
INSIGHT_IMPORTANCE = 9  # 记忆被坦白/纠正之后新增的记忆的重要度，跟 agent.py 的 IMPORTANCE_REFLECTION 同一个用意
TURN_IMPORTANCE = 4  # 玩家的问话、NPC 自己的回答，这类对话记忆的重要度，跟 agent.py 的 IMPORTANCE_SAID 同一量级
RECALL_K = 4  # 每轮回复前，从记忆里挑几条相关的塞进提示词


@dataclass(frozen=True)
class Recollection:
    """NPC 记忆里关于案发当晚的一条独立说法，不是整段真相，是拆开的碎片——这是"罗生门"
    式设计的核心：真相不会被完整打包塞给某一个人，要靠玩家在多个人、多条不完整甚至
    有偏差的说法之间找矛盾、找印证，自己拼出来。"""

    fact_id: str  # 这条记忆对应哪个案情节点，不同 NPC 关于同一 fact_id 的记忆可能互相印证、也可能矛盾
    text_before: str  # 状态转变之前会说的话
    accuracy: str  # "witnessed"（亲历且准确）| "distorted"（亲历但自己理解有偏差）| "secondhand"（听来的，可能走样）
    withheld: bool = False  # True：一开始藏着，得先满足 transition_condition 才会说出口；
    # 跟 accuracy 是两个独立维度——一条 witnessed（内容准确）的记忆完全可以同时是 withheld（故意不说）
    text_after: str | None = None  # 状态转变之后该说的话。withheld 类：这才是真相；distorted/secondhand
    # 类：变成带着"我可能记错/听错了"这种不确定口吻，不会凭空编出正确答案
    transition_condition: str | None = None  # 玩家要做到什么，这条记忆会发生上面那种转变——留 None
    # 表示这条记忆状态固定、永远说 text_before（这版剧本里暂时用不到，但结构上留着这个口子）


@dataclass(frozen=True)
class Backstory:
    """跟案情无关的背景记忆，只是给 NPC 垫一点日常细节，也给 MemoryStore 的检索机制一点
    "不是所有记忆都同等重要"的真实场景——不然每个 NPC 脑子里就只有一条秘密，重要度/
    相关度排序毫无意义。"""

    text: str
    importance: int = 3


@dataclass(frozen=True)
class CaseNPC:
    id: str
    name: str
    role: str  # 身份，一句话
    persona: str  # 性格/说话习惯
    recollections: tuple[Recollection, ...]
    backstory: tuple[Backstory, ...] = ()


@dataclass(frozen=True)
class Ending:
    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Case:
    title: str
    room: str  # 房间描述，开场白用
    background: str  # 给玩家看的案情简介
    npcs: tuple[CaseNPC, ...]

    def npc(self, npc_id: str) -> CaseNPC:
        for n in self.npcs:
            if n.id == npc_id:
                return n
        raise KeyError(npc_id)


TRUTH_ENDING = Ending(
    id="truth",
    title="真相大白：她是自己选择离开的",
    text=(
        "沈微没有被害，也没有卷款潜逃——她是自己选择走的。账目上的窟窿，是她自己垫钱帮图书馆"
        "度过难关，却被老周误会成挪用；表哥上门讨债、阿May的流言，让她在小镇上越来越喘不过气；"
        "恰好这时外地有了新的机会，她趁老马那晚脱岗、没人注意，独自离开，只告诉了小雨一个人。"
        "案件真相大白，几个人也各自明白了自己无意间造成了什么，也发现自己这几天说的话里，"
        "有几处其实是自己记错了、或者以讹传讹传歪了的。"
    ),
)
WRONG_ACCUSATION_ENDING = Ending(
    id="wrong_accusation",
    title="错怪无辜：案件表面结案，真相仍是个谜",
    text=(
        "你没能问出沈微离开的真正原因，只拼凑出账目异常、表哥讨债这些看似可疑的线索，"
        "把嫌疑压在了错的人身上结了案。镇长满意地宣布案件了结，但沈微到底去了哪里、为什么走，"
        "始终没人真正知道——而你手上那些线索里，有几条其实是别人记错、听岔了的，你没来得及"
        "去交叉核实，就当成了定论。"
    ),
)
UNRESOLVED_ENDING = Ending(
    id="unresolved",
    title="悬而未决：线索太少，案件不了了之",
    text="问话时间到了，你手上的线索太零散，拼不出任何站得住脚的结论。案件被暂时搁置，沈微依旧下落不明。",
)

# 会让人往错的方向猜的线索：单独问出这些、却没问出关键的 departure，说明查偏了方向
MISLEADING_FACTS = frozenset({"ledger", "debt_fight"})


def resolve_ending(discovered: frozenset[str]) -> Ending:
    """根据玩家实际问出来的事实集合判定结局，规则是覆盖优先级：
    问出了关键的"自愿离开"线索（departure）——不管还问出了别的什么——都是真相大白；
    没问出 departure，但问出了至少一条容易误导人往错方向猜的线索——错怪无辜；
    以上都不满足——线索太少，不了了之。"""
    if "departure" in discovered:
        return TRUTH_ENDING
    if discovered & MISLEADING_FACTS:
        return WRONG_ACCUSATION_ENDING
    return UNRESOLVED_ENDING


MISSING_PERSON_CASE = Case(
    title="失踪的图书馆管理员",
    room="镇立图书馆后堂——一间不大的储藏室兼员工休息室，堆着旧书和纸箱，只有一盏灯亮着。",
    background=(
        "图书馆兼职管理员沈微，三天前闭馆后再也没有出现过。镇长把知道内情的几个人都叫到了后堂，"
        "请你——外请的调查员——问清楚沈微到底出了什么事。这几个人各自记得的都只是一部分，"
        "有的还记岔了，你得在他们的说法里找矛盾、找印证，自己拼出真相。"
    ),
    npcs=(
        CaseNPC(
            id="zhou",
            name="老周",
            role="图书馆馆长，沈微的上司",
            persona="严肃、护短，把图书馆的名声看得很重，说话喜欢先摆事实、后表态度，一被戳到软肋容易语气变硬。",
            backstory=(Backstory("上周镇长来图书馆借了本地方志，还念叨着下次多进点新书。"),),
            recollections=(
                Recollection(
                    fact_id="ledger",
                    text_before="账上有一笔钱对不上，我一直怀疑是沈微私自拿的，说不定她就是心虚才跑了。",
                    accuracy="distorted",
                    text_after="……你这么一说，我倒犹豫了，会不会我一直都想岔了，那笔钱另有说法？",
                    transition_condition="玩家出示了矛盾的证据——比如提到沈微其实是自己垫钱补窟窿、或者从别人那"
                    "问出账目异常另有隐情——来质疑老周这个「挪用」的判断",
                ),
            ),
        ),
        CaseNPC(
            id="xiaoyu",
            name="小雨",
            role="沈微最好的朋友，图书馆常客",
            persona="情绪化、心直口快，很担心沈微，一开口就容易往最坏的方向猜，但其实心里藏着沈微亲口交代要保密的事。",
            backstory=(Backstory("沈微上个月推荐她看了本推理小说，她还没看完，念叨了好几次。"),),
            recollections=(
                Recollection(
                    fact_id="departure",
                    text_before="我们什么都聊，但她最近真没提过要走或者有什么麻烦，我也想不通到底发生了什么。",
                    accuracy="witnessed",
                    withheld=True,
                    text_after="她最近一直在偷偷联系外地一个人，好像是有新的工作机会，让我千万别说出去，"
                    "连她家里人都不知道……我现在也不知道该不该说了。",
                    transition_condition="玩家提到手机、信件、车票、外地、新工作、联系人这类线索，或者玩家已经"
                    "表现出对小雨足够的信任（比如先跟她分享了调查进展、明确说会保护她不受牵连），或者已经反复追问超过两次",
                ),
                Recollection(
                    fact_id="debt_fight",
                    text_before="我隐约听说表哥好像来找过她，好像是为了钱的事，具体我也没亲眼看见，说不清楚。",
                    accuracy="secondhand",
                    text_after="这事儿我确实是听说的，具体细节我也说不准，你要真想弄清楚，还是得问当事人。",
                    transition_condition="玩家提到已经从陈刚本人那问出来的、具体的版本，拿这个去跟小雨这条道听途说的说法对质",
                ),
            ),
        ),
        CaseNPC(
            id="laoma",
            name="老马",
            role="图书馆保安，失踪当晚值夜班",
            persona="木讷寡言，说话慢半拍，能用一个字回答绝不用两个字，一旦被戳穿会明显慌乱、开始语无伦次。",
            backstory=(Backstory("上周他还念叨着想换双巡逻穿的鞋，说现在这双磨脚。", importance=2),),
            recollections=(
                Recollection(
                    fact_id="guard_absent",
                    text_before="我一直在岗位上，正常巡逻，啥异常都没看见。",
                    accuracy="witnessed",
                    withheld=True,
                    text_after="……那晚我，我喝了点酒，迷糊了大概半个钟头，后堂那边我确实没盯着。我怕说出来丢了"
                    "这份工作，就没敢说。",
                    transition_condition="玩家提到监控、岗位记录、脱岗、喝酒这类具体证据，或者玩家已经从别人那"
                    "问出「老马那晚可能不在岗」并拿这个去质问他",
                ),
            ),
        ),
        CaseNPC(
            id="chengang",
            name="陈刚",
            role="沈微的表哥",
            persona="焦躁、防备心重，说话时会下意识撇清关系，容易被逼问急了就露出破绽。",
            backstory=(Backstory("他其实挺喜欢沈微做的手工书签，收了好几个，只是不好意思说。"),),
            recollections=(
                Recollection(
                    fact_id="debt_fight",
                    text_before="我根本没来过图书馆，那晚我在家，跟这事一点关系没有。",
                    accuracy="witnessed",
                    withheld=True,
                    text_after="行，我说——沈微欠我一大笔钱，那晚我是来找她要账的，我们吵了一架，我说了些难听话，"
                    "但我没碰她，说完我自己就走了。",
                    transition_condition="玩家提到目击证词、有人当晚见过一个男的来找沈微、债务、借条这类线索",
                ),
                Recollection(
                    fact_id="rumor",
                    text_before="我倒是听人说过，她好像跟镇上什么人走得挺近，具体是谁我也没细问。",
                    accuracy="secondhand",
                    text_after="这事儿我也是听来的，谁传的我可说不准，你要较真还是得自己去查源头。",
                    transition_condition="玩家已经从阿May那问出谣言真正的源头/版本，拿这个去跟陈刚这条道听途说的说法对质",
                ),
            ),
        ),
        CaseNPC(
            id="amay",
            name="阿May",
            role="镇长助理、图书馆志愿者",
            persona="热心、话多，看起来什么都愿意帮忙打听，实际上很会把话题往别人身上带，被点破谎言源头时会明显心虚、语速变快。",
            backstory=(Backstory("上周她还张罗着给图书馆办了个读书角活动，来了不少人。"),),
            recollections=(
                Recollection(
                    fact_id="rumor",
                    text_before="我就是随口帮忙问问，真要说谁信得过、谁信不过，我可说不准，你问问老周或者陈刚吧。",
                    accuracy="witnessed",
                    withheld=True,
                    text_after="……那些话是我说的。我就是看她人缘好，嘴一快传了几句不该传的话，没想到会传成那样，"
                    "给她添了那么大压力，我不是故意的。",
                    transition_condition="玩家从至少两个不同的人那听到同一个谣言的不同版本，并拿这个去质问阿May，"
                    "或者直接指出谣言的源头是她",
                ),
            ),
        ),
    ),
)


class Reply(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_REPLY_CHARS, description="要说的一句话")
    mentions: list[str] = Field(
        default_factory=list,
        description="这句话实际谈到了自己脑子里哪些记忆对应的 fact_id（提示词里每条记忆前面标了 [fact_id=...]），"
        "没谈到任何一条就留空列表",
    )


class TransitionJudgement(BaseModel):
    """对这个 NPC 身上每条还没转变状态的记忆，依次判断玩家这句话（结合对话历史）有没有让转变发生。
    两种情况：知道但藏着的记忆——玩家问到点子上/出示了线索/建立了信任，才算条件成立；
    自己可能记错/听错的记忆——玩家明确拿出别处印证到的矛盾信息来对质，才算条件成立，
    单纯的质疑语气不算。"""

    transitioned: list[bool] = Field(description="每条记忆是否发生转变，按输入顺序一一对应")


REPLY_TOOL = {"name": "reply", "description": "说一句回应玩家的话", "parameters": Reply.model_json_schema()}
TRANSITION_TOOL = {
    "name": "judge_transition",
    "description": "判断玩家这句话有没有让 NPC 身上每条还没转变状态的记忆发生转变",
    "parameters": TransitionJudgement.model_json_schema(),
}


@dataclass
class _NpcState:
    transitioned: set[str] = field(default_factory=set)  # 已经发生转变的 fact_id
    store: MemoryStore = field(default_factory=MemoryStore)


class CaseSession:
    """一局案件的运行状态：谁问出了什么、谁还瞒着什么、谁的哪条记忆已经被质疑过。跟 Agent
    不共享任何状态——案件模式是独立的一局游戏，不是小镇沙盒的一部分，两者可以在同一个
    进程里并存。每个 NPC 有一个真正的 townmind.memory.MemoryStore：案情相关的记忆、
    背景记忆、玩家的问话、NPC 自己的回答，全都作为普通 Memory 条目存进去，靠新近度+
    重要度+（配了 embedder 时）语义相关度这套公式决定每轮该想起哪些。"""

    def __init__(self, case: Case, llm: LLMClient | None, embedder=None, max_turns: int = 30):
        self.case = case
        self.llm = llm
        self.embedder = embedder  # 可选：townmind.llm.embeddings.OpenAIEmbedder；没配时相关度退化成旧公式
        self.max_turns = max_turns
        self.turns = 0
        self._base_time = _time.time() - BACKSTORY_AGE
        self._states: dict[str, _NpcState] = {}
        for n in case.npcs:
            st = _NpcState()
            for b in n.backstory:
                st.store.add(b.text, b.importance, self._base_time)
            for r in n.recollections:
                st.store.add(r.text_before, CASE_IMPORTANCE, self._base_time)
            self._states[n.id] = st
        self.discovered: set[str] = set()
        self.evidence: dict[str, list[dict]] = defaultdict(list)  # fact_id -> 玩家实际听到过的各方说法，证据板用
        self.ended = False
        self.stats: dict[str, int] = defaultdict(int)

    def _now(self) -> float:
        # 问话本身也让时间往前走一点（一轮一个单位），这样"刚问的""刚答的"这类记忆的新近度
        # 分数会明显比案发当晚（3 天前）的旧记忆高——不然所有记忆时间戳都差不多，新近度这个
        # 维度就没有区分度了
        return self._base_time + BACKSTORY_AGE + self.turns

    def npc_status(self, npc_id: str) -> dict:
        n = self.case.npc(npc_id)
        st = self._states[npc_id]
        return {
            "name": n.name,
            "role": n.role,
            "transitioned": sorted(st.transitioned),
            "pending": sorted(
                r.fact_id for r in n.recollections if r.transition_condition and r.fact_id not in st.transitioned
            ),
        }

    def evidence_board(self) -> dict[str, list[dict]]:
        """按 fact_id 分组、玩家实际听到过的各方说法——同一个 fact_id 底下如果有多条、
        内容还不一样，就是"罗生门"要玩家自己去比对的地方。"""
        return dict(self.evidence)

    def time_left(self) -> int:
        return max(0, self.max_turns - self.turns)

    async def talk(self, npc_id: str, player_text: str) -> str:
        """玩家跟某个 NPC 说一句话，返回 NPC 的回应。ended=True 之后不再接受新的问话。"""
        if self.ended:
            return "（问话时间已经结束，案件到此为止。）"
        npc = self.case.npc(npc_id)
        st = self._states[npc_id]

        res = safety.check_player_text(player_text)
        self.stats["player_turns"] += 1
        for f in res.flags:
            self.stats[f"player_{f}"] += 1
        if not res.ok:
            return "（……ta 没听清你说了什么。）"
        text = res.text

        self.turns += 1
        now = self._now()
        st.store.add(f"玩家问你：「{text}」", TURN_IMPORTANCE, now)

        pending = [
            r for r in npc.recollections if r.transition_condition and r.fact_id not in st.transitioned
        ]
        if pending and self.llm is not None:
            for fact_id in await self._judge_transitions(npc, st, pending, text):
                st.transitioned.add(fact_id)
                rec = next(r for r in npc.recollections if r.fact_id == fact_id)
                if rec.text_after:
                    st.store.add(rec.text_after, INSIGHT_IMPORTANCE, now)
                if rec.accuracy != "witnessed":
                    # 纠正了一条不准的记忆，额外存一条反思——直接复刻 agent.py
                    # _learn_from_correction 的思路：不只是这一轮起效，也留一条更高重要度的
                    # "我可能记错了"记忆，以后聊到别的话题时语义检索也可能把它翻出来
                    st.store.add(f"你意识到自己关于这件事的说法可能记错/听错了", INSIGHT_IMPORTANCE, now)
                self.stats["transitions"] += 1

        reply_text, mentions = await self._generate_reply(npc, st, text, now)
        st.store.add(f"你回答说：「{reply_text}」", TURN_IMPORTANCE, now)

        for fact_id in mentions:
            rec = next((r for r in npc.recollections if r.fact_id == fact_id), None)
            if rec is None:  # 大模型报了个不存在的 fact_id，不让脏数据进证据板
                continue
            transitioned = fact_id in st.transitioned
            current_text = rec.text_after if (transitioned and rec.text_after) else rec.text_before
            self.evidence[fact_id].append(
                {"npc": npc.name, "text": current_text, "accuracy": rec.accuracy,
                 "state": "transitioned" if transitioned else "original"}
            )
            self.discovered.add(fact_id)

        if self.turns >= self.max_turns:
            self.ended = True
        return reply_text

    def _recent_history(self, st: _NpcState, k: int = 8) -> str:
        recent = sorted(st.store.memories, key=lambda m: m.time, reverse=True)[:k]
        recent.reverse()
        return "\n".join(m.text for m in recent)

    async def _judge_transitions(self, npc: CaseNPC, st: _NpcState, pending: list[Recollection], player_text: str) -> list[str]:
        system = (
            f"你在判断一个案件游戏里的 NPC「{npc.name}」（{npc.role}）身上，每条还没转变状态的记忆，"
            "玩家最新这句话（结合下面的对话历史）有没有让转变发生。两种情况：\n"
            "1. 这条记忆是「知道但藏着」的：玩家问到点子上、出示了对应线索、或者已经建立了足够信任，才算条件成立；\n"
            "2. 这条记忆本身「可能不准」（NPC 自己深信不疑）：只有玩家明确拿出别处印证到的、矛盾的具体信息来"
            "对质，才算条件成立——玩家单纯的质疑语气、反问，但没提供具体矛盾信息，不算。"
        )
        history = self._recent_history(st)
        items_desc = "\n".join(
            f"{i + 1}. [{r.accuracy}{'，藏着' if r.withheld else ''}] 转变条件：{r.transition_condition}"
            for i, r in enumerate(pending)
        )
        user = f"对话历史：\n{history}\n\n玩家最新这句话：「{player_text}」\n\n待判断的记忆：\n{items_desc}"
        call = await self._call_tool(system, user, TRANSITION_TOOL, "transition_judge_calls")
        if call is None:
            return []
        try:
            flags = TransitionJudgement(**call.arguments).transitioned
        except Exception as e:
            log.warning("[case] 转变判定格式不对（%s: %s），这一轮当成都没转变", type(e).__name__, e)
            return []
        if len(flags) != len(pending):
            log.warning("[case] 转变判定数量（%d）跟待判断记忆数量（%d）对不上，跳过", len(flags), len(pending))
            return []
        return [r.fact_id for r, ok in zip(pending, flags) if ok]

    async def _generate_reply(self, npc: CaseNPC, st: _NpcState, player_text: str, now: float) -> tuple[str, list[str]]:
        if self.llm is None:
            return self._fallback_reply(npc, st), []
        system = self._build_prompt(npc, st)
        query_embedding = await self._embed_query(player_text)
        recalled = st.store.recall(frozenset(), now, k=RECALL_K, query_embedding=query_embedding)
        recalled_text = "\n".join(f"- {format_age(now - m.time)}：{m.text}" for m in recalled)
        history = self._recent_history(st)
        user = (
            f"你想起的相关记忆：\n{recalled_text}\n\n到目前为止的对话：\n{history}\n\n"
            "请决定你现在要说的这句话。"
        )
        call = await self._call_tool(system, user, REPLY_TOOL, "reply_calls")
        if call is None:
            return self._fallback_reply(npc, st), []
        try:
            reply = Reply(**call.arguments)
        except Exception as e:
            log.warning("[case] 回应格式不对（%s: %s），退回兜底台词", type(e).__name__, e)
            return self._fallback_reply(npc, st), []
        ok, flags = self._check_output(reply.text)
        if not ok:
            self.stats["guard_blocked"] += 1
            log.warning("[case][%s] output blocked %s: %s", npc.id, flags, reply.text)
            return self._fallback_reply(npc, st), []
        valid_ids = {r.fact_id for r in npc.recollections}
        mentions = [f for f in reply.mentions if f in valid_ids]
        return reply.text, mentions

    def _build_prompt(self, npc: CaseNPC, st: _NpcState) -> str:
        parts = [
            f"你是案件游戏里的 NPC「{npc.name}」（{npc.role}）。{npc.persona}",
            f"案发地点：{self.case.room}",
            "每次只说一句话，不超过 40 个字，要符合你的性格；不要用旁白或者动作描写，只说台词本身。",
            "玩家是被请来查这件事的调查员，你在被问话，不是在自由聊天，回答要跟问题相关。",
            "以下是你脑子里关于这件事的记忆，每条现在是什么状态已经由系统判定好了，你只需要照着这个状态说：",
        ]
        for r in npc.recollections:
            transitioned = r.fact_id in st.transitioned
            if r.withheld and not transitioned:
                parts.append(
                    f"- [fact_id={r.fact_id}] 你其实知道真相，但还没到说的时候，先这样搪塞：{r.text_before}"
                    "，口吻要自然，不要表现得心虚。"
                )
            elif r.withheld and transitioned:
                parts.append(f"- [fact_id={r.fact_id}] 条件已经成立，你现在必须老实说：{r.text_after}")
            elif not transitioned:
                hint = "（其实你可能记错/听错了，但你自己完全没意识到，说的时候要显得很确定）" if r.accuracy != "witnessed" else ""
                parts.append(f"- [fact_id={r.fact_id}] 你确信这是事实，自信地说：{r.text_before}{hint}")
            else:
                parts.append(
                    f"- [fact_id={r.fact_id}] 刚有人拿矛盾的信息指出你这条记忆可能有问题，你现在应该带着"
                    f"不确定的口吻说：{r.text_after}，不要凭空编出一个新的确切答案，老实承认自己可能记错/听错了就行。"
                )
        parts.append(
            "如果你说的内容确实涉及上面列出的某条记忆（哪怕只是部分），在 mentions 字段里填上对应的 fact_id；"
            "没涉及任何一条就留空列表。玩家问的事如果跟这些都不相关，就正常地、符合你性格地回应，"
            "可以聊聊你脑子里想起的别的相关记忆。"
        )
        parts.append("不要主动承认自己在说谎或记错、不要提到「fact_id」「系统判定」这些词——这些是只有你知道的内心状态，不是台词。")
        return "\n".join(parts)

    def _check_output(self, text: str) -> tuple[bool, list[str]]:
        """案件模式的出口检查：沿用小镇那套"出戏/泄露提示词"检测和长度上限，但不用
        ungrounded_items——那是给小镇食物/物品设定配的词表，案件是完全不同的世界观，
        硬套只会把"债务""流言"这些正常剧情词也一起误伤（跟阿尔伯塔大学是同一类问题）。"""
        flags: list[str] = []
        if not text or not text.strip():
            flags.append("empty")
        elif len(text) > MAX_REPLY_CHARS:
            flags.append("too_long")
        if safety._LEAK_OR_OOC.search(text):
            flags.append("leak_or_out_of_character")
        return not flags, flags

    def _fallback_reply(self, npc: CaseNPC, st: _NpcState) -> str:
        for r in npc.recollections:
            transitioned = r.fact_id in st.transitioned
            return r.text_after if (transitioned and r.text_after) else r.text_before
        return "……"

    async def _call_tool(self, system: str, user: str, tool: dict, stat_key: str):
        self.stats[stat_key] += 1
        try:
            return await self.llm.choose_tool(system, user, [tool])
        except Exception as e:
            log.warning("[case] LLM 调用异常（%s: %s）", type(e).__name__, e)
            return None

    async def _embed_query(self, player_text: str) -> list[float] | None:
        """把玩家这句话变成向量，用来跟 NPC 脑子里每条记忆比相似度——没配 embedder、
        调用失败、或者返回数量不对，都优雅退化成 None（MemoryStore.score 据此退回不带
        语义相关度的旧公式），不让这一层的问题拖垮决策主流程。跟 agent.py 里
        Agent._embed_query 是同一个退化哲学。"""
        if self.embedder is None or not player_text:
            return None
        try:
            vectors = await self.embedder.embed([player_text])
        except Exception as e:
            log.warning("[case] embedding 调用异常（%s: %s），本轮相关度退化成旧公式", type(e).__name__, e)
            return None
        if len(vectors) != 1:
            return None
        return vectors[0]

    def resolve(self) -> Ending:
        return resolve_ending(frozenset(self.discovered))
