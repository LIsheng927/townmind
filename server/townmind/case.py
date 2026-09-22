"""案件模式：把 Agent 的对话能力用在一个有明确目标的场景里——玩家扮演调查员，
在限定的房间里跟几个 NPC 问话，NPC 各自藏着一部分真相，默认用谎言/掩饰说法搪塞，
只有玩家问到点子上（出示对应线索，或者从别的 NPC 那换到情报）才会松口说实话。
最终案件的结局，由玩家实际问出来的这些"事实"的集合决定。

跟 townmind/agent.py 里的小镇 Agent 是两套不同的东西，不复用它的工具调用/移动/记忆检索
那一整套——那套是为"NPC 自己会走会闲逛"这种沙盒设计的，这里的场景是"几个人固定待在
同一个房间里，一问一答"，用不上 go_to/follow_player/语义记忆检索这些。复用的是更底层、
跟"是小镇还是案件"无关的东西：LLMClient.choose_tool 接口、safety.check_player_text
入口检查、safety._LEAK_OR_OOC 出戏/泄密检查。

一次问话分两步，都是强制工具调用（跟 agent.py 的风格一致，不用自由文本生成，避免大模型
不按格式说话导致解析失败）：
  1. 坦白判定：这个 NPC 身上还没坦白的每条秘密，让大模型结合对话历史判断玩家这句话有没有
     让坦白条件成立——成立了，这条秘密从这一轮起变成"已坦白"，状态只会从"瞒"变成"说"，
     不会说漏嘴之后又反悔变回"瞒"（不然同一句台词可能前后矛盾）。
  2. 生成回应：根据"这条秘密现在是坦白还是隐瞒"的状态生成台词——隐瞒的部分用当事人预设的
     掩饰说法（要求口吻自然，不是那种一戳就崩的心虚"死鸭子嘴硬"），已经坦白的部分才允许说实情。

"限时 15 分钟"这版先简化成"最多问这么多轮"（每跟任意一个 NPC 说一句话算一轮），不追踪
真实墙钟时间——demo/测试阶段更好控制和复现，以后接 Unity 时可以再换成真实计时器，
到点直接调用 resolve()，逻辑不用变。
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from . import safety
from .llm.base import LLMClient

log = logging.getLogger("townmind.case")

MAX_REPLY_CHARS = 80  # 案件里 NPC 交代缘由有时候比小镇闲聊台词长，给宽松些的硬上限（提示词里仍要求不超过 40 字）


@dataclass(frozen=True)
class Secret:
    """NPC 身上藏着的一条真相。"""

    fact_id: str  # 结局判定用这个 id 认事实，同一个案件里必须唯一
    summary: str  # 一句话描述这条事实是什么（给状态面板/结局判定用，不直接进提示词）
    truth: str  # 坦白之后 NPC 该说出的实情内容
    cover_story: str  # 默认会说的掩饰说法/谎言
    reveal_condition: str  # 玩家要做到什么，NPC 才会松口——自然语言描述，给大模型判定用


@dataclass(frozen=True)
class CaseNPC:
    id: str
    name: str
    role: str  # 身份，一句话
    persona: str  # 性格/说话习惯
    secrets: tuple[Secret, ...]


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
        "案件真相大白，几个人也各自明白了自己无意间造成了什么。"
    ),
)
WRONG_ACCUSATION_ENDING = Ending(
    id="wrong_accusation",
    title="错怪无辜：案件表面结案，真相仍是个谜",
    text=(
        "你没能问出沈微离开的真正原因，只拼凑出账目异常、表哥讨债这些看似可疑的线索，"
        "把嫌疑压在了错的人身上结了案。镇长满意地宣布案件了结，但沈微到底去了哪里、为什么走，"
        "始终没人真正知道。"
    ),
)
UNRESOLVED_ENDING = Ending(
    id="unresolved",
    title="悬而未决：线索太少，案件不了了之",
    text="问话时间到了，你手上的线索太零散，拼不出任何站得住脚的结论。案件被暂时搁置，沈微依旧下落不明。",
)

# 会让人往错的方向猜的线索：单独问出这些、却没问出关键的 job_offer，说明查偏了方向
MISLEADING_FACTS = frozenset({"ledger", "debt_fight"})


def resolve_ending(discovered: frozenset[str]) -> Ending:
    """根据玩家实际问出来的事实集合判定结局，规则很直接，就是覆盖优先级：
    问出了关键的"自愿离开"线索（job_offer）——不管还问出了别的什么——都是真相大白；
    没问出 job_offer，但问出了至少一条容易误导人往错方向猜的线索（账目异常/讨债）——错怪无辜；
    以上都不满足——线索太少，不了了之。"""
    if "job_offer" in discovered:
        return TRUTH_ENDING
    if discovered & MISLEADING_FACTS:
        return WRONG_ACCUSATION_ENDING
    return UNRESOLVED_ENDING


MISSING_PERSON_CASE = Case(
    title="失踪的图书馆管理员",
    room="镇立图书馆后堂——一间不大的储藏室兼员工休息室，堆着旧书和纸箱，只有一盏灯亮着。",
    background=(
        "图书馆兼职管理员沈微，三天前闭馆后再也没有出现过。镇长把知道内情的几个人都叫到了后堂，"
        "请你——外请的调查员——问清楚沈微到底出了什么事。你可以自由提问，也可以把从一个人那问出来的"
        "线索去问另一个人。"
    ),
    npcs=(
        CaseNPC(
            id="zhou",
            name="老周",
            role="图书馆馆长，沈微的上司",
            persona="严肃、护短，把图书馆的名声看得很重，说话喜欢先摆事实、后表态度，一被戳到软肋容易语气变硬。",
            secrets=(
                Secret(
                    fact_id="ledger",
                    summary="老周发现账目对不上，怀疑沈微挪用了一笔捐款，一直没声张",
                    truth="账上有一笔钱对不上，我怀疑是沈微动的，本想私下问清楚再说，没想到她就这么不见了——我怕说出来大家先入为主，把她当贼。",
                    cover_story="图书馆账目一直清清楚楚，没有任何异常，我跟沈微相处得很好，实在想不通她为什么会不见。",
                    reveal_condition="玩家提到账本、账目、财务、捐款、钱对不上这类线索，或者直接质问老周账目是不是有问题",
                ),
            ),
        ),
        CaseNPC(
            id="xiaoyu",
            name="小雨",
            role="沈微最好的朋友，图书馆常客",
            persona="情绪化、心直口快，很担心沈微，一开口就容易往最坏的方向猜，但其实心里藏着沈微亲口交代要保密的事。",
            secrets=(
                Secret(
                    fact_id="job_offer",
                    summary="沈微最近在偷偷联系外地的一个机会，打算离开小镇，让小雨保密",
                    truth="她最近一直在偷偷联系外地一个人，好像是有新的工作机会，让我千万别说出去，连她家里人都不知道……我现在也不知道该不该说了。",
                    cover_story="我们什么都聊，但她最近真没提过要走或者有什么麻烦，我也想不通到底发生了什么。",
                    reveal_condition="玩家提到手机、信件、车票、外地、新工作、联系人这类线索，或者玩家已经表现出对小雨足够的信任"
                    "（比如先跟她分享了调查进展、明确说会保护她不受牵连），或者已经反复追问超过两次",
                ),
            ),
        ),
        CaseNPC(
            id="laoma",
            name="老马",
            role="图书馆保安，失踪当晚值夜班",
            persona="木讷寡言，说话慢半拍，能用一个字回答绝不用两个字，一旦被戳穿会明显慌乱、开始语无伦次。",
            secrets=(
                Secret(
                    fact_id="guard_absent",
                    summary="老马那晚脱岗喝酒，后堂门口有半小时没人看守",
                    truth="……那晚我，我喝了点酒，迷糊了大概半个钟头，后堂那边我确实没盯着。我怕说出来丢了这份工作，就没敢说。",
                    cover_story="我一直在岗位上，正常巡逻，啥异常都没看见。",
                    reveal_condition="玩家提到监控、岗位记录、脱岗、喝酒这类具体证据，或者玩家已经从别人那问出「老马那晚可能不在岗」并拿这个去质问他",
                ),
            ),
        ),
        CaseNPC(
            id="chengang",
            name="陈刚",
            role="沈微的表哥",
            persona="焦躁、防备心重，说话时会下意识撇清关系，容易被逼问急了就露出破绽。",
            secrets=(
                Secret(
                    fact_id="debt_fight",
                    summary="陈刚那晚来找沈微讨债，两人大吵一架，之后陈刚自己先走了",
                    truth="行，我说——沈微欠我一大笔钱，那晚我是来找她要账的，我们吵了一架，我说了些难听话，但我没碰她，说完我自己就走了。",
                    cover_story="我根本没来过图书馆，那晚我在家，跟这事一点关系没有。",
                    reveal_condition="玩家提到目击证词、有人当晚见过一个男的来找沈微、债务、借条这类线索",
                ),
            ),
        ),
        CaseNPC(
            id="amay",
            name="阿May",
            role="镇长助理、图书馆志愿者",
            persona="热心、话多，看起来什么都愿意帮忙打听，实际上很会把话题往别人身上带，被点破谎言源头时会明显心虚、语速变快。",
            secrets=(
                Secret(
                    fact_id="rumor",
                    summary="阿May出于嫉妒散布过针对沈微的谣言，这是沈微压力的来源之一",
                    truth="……那些话是我说的。我就是看她人缘好，嘴一快传了几句不该传的话，没想到会传成那样，给她添了那么大压力，我不是故意的。",
                    cover_story="我就是随口帮忙问问，真要说谁信得过、谁信不过，我可说不准，你问问老周或者陈刚吧。",
                    reveal_condition="玩家从至少两个不同的人那听到同一个谣言的不同版本，并拿这个去质问阿May，或者直接指出谣言的源头是她",
                ),
            ),
        ),
    ),
)


class Reply(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_REPLY_CHARS, description="要说的一句话")


class RevealJudgement(BaseModel):
    """对这个 NPC 身上每条还没坦白的秘密，依次判断玩家这句话（结合对话历史）有没有让坦白条件成立。"""

    revealed: list[bool] = Field(description="每条秘密的坦白条件是否已经成立，按输入顺序一一对应")


REPLY_TOOL = {"name": "reply", "description": "说一句回应玩家的话", "parameters": Reply.model_json_schema()}
REVEAL_JUDGEMENT_TOOL = {
    "name": "judge_reveal",
    "description": "判断玩家这句话有没有让 NPC 身上每条还没坦白的秘密的坦白条件成立",
    "parameters": RevealJudgement.model_json_schema(),
}


@dataclass
class _NpcState:
    revealed: set[str] = field(default_factory=set)  # 这个 NPC 身上已经坦白的 fact_id
    transcript: list[tuple[str, str]] = field(default_factory=list)  # (speaker, text)


class CaseSession:
    """一局案件的运行状态：谁问出了什么、谁还瞒着什么。跟 Agent 不共享任何状态——
    案件模式是独立的一局游戏，不是小镇沙盒的一部分，两者可以在同一个进程里并存。"""

    def __init__(self, case: Case, llm: LLMClient | None, max_turns: int = 30):
        self.case = case
        self.llm = llm
        self.max_turns = max_turns
        self.turns = 0
        self._states: dict[str, _NpcState] = {n.id: _NpcState() for n in case.npcs}
        self.discovered: set[str] = set()
        self.ended = False
        self.stats: dict[str, int] = defaultdict(int)

    def npc_status(self, npc_id: str) -> dict:
        n = self.case.npc(npc_id)
        st = self._states[npc_id]
        return {
            "name": n.name,
            "role": n.role,
            "revealed": sorted(st.revealed),
            "hidden": sorted(s.fact_id for s in n.secrets if s.fact_id not in st.revealed),
        }

    def time_left(self) -> int:
        return max(0, self.max_turns - self.turns)

    async def talk(self, npc_id: str, player_text: str) -> str:
        """玩家跟某个 NPC 说一句话，返回 NPC 的回应。ended=True 之后不再接受新的问话——
        调用方（demo/Unity）该看 self.ended 决定要不要结案。"""
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
        st.transcript.append(("玩家", text))

        hidden = [s for s in npc.secrets if s.fact_id not in st.revealed]
        if hidden and self.llm is not None:
            for fact_id in await self._judge_reveal(npc, st, hidden):
                st.revealed.add(fact_id)
                self.discovered.add(fact_id)
                self.stats["facts_revealed"] += 1

        reply_text = await self._generate_reply(npc, st)
        st.transcript.append((npc.name, reply_text))

        if self.turns >= self.max_turns:
            self.ended = True
        return reply_text

    async def _judge_reveal(self, npc: CaseNPC, st: _NpcState, hidden: list[Secret]) -> list[str]:
        system = (
            f"你在判断一个案件游戏里的 NPC「{npc.name}」（{npc.role}）身上还没坦白的秘密，"
            "玩家最新这句话（结合下面的对话历史），有没有让每条秘密各自的坦白条件成立。"
            "只有真的满足条件才算成立，玩家随口一问、或者恰好猜中了但没有说出关键线索，不算。"
        )
        history = "\n".join(f"{spk}：{t}" for spk, t in st.transcript[-10:])
        secrets_desc = "\n".join(f"{i + 1}. 坦白条件：{s.reveal_condition}" for i, s in enumerate(hidden))
        user = f"对话历史：\n{history}\n\n待判断的秘密：\n{secrets_desc}"
        call = await self._call_tool(system, user, REVEAL_JUDGEMENT_TOOL, "reveal_judge_calls")
        if call is None:
            return []
        try:
            flags = RevealJudgement(**call.arguments).revealed
        except Exception as e:
            log.warning("[case] 坦白判定格式不对（%s: %s），这一轮当成都没成立", type(e).__name__, e)
            return []
        if len(flags) != len(hidden):
            log.warning("[case] 坦白判定数量（%d）跟待判断秘密数量（%d）对不上，跳过", len(flags), len(hidden))
            return []
        return [s.fact_id for s, ok in zip(hidden, flags) if ok]

    async def _generate_reply(self, npc: CaseNPC, st: _NpcState) -> str:
        if self.llm is None:
            return self._fallback_reply(npc, st)
        system = self._build_prompt(npc, st)
        history = "\n".join(f"{spk}：{t}" for spk, t in st.transcript[-10:])
        user = f"到目前为止的对话：\n{history}\n\n请决定你现在要说的这句话。"
        call = await self._call_tool(system, user, REPLY_TOOL, "reply_calls")
        if call is None:
            return self._fallback_reply(npc, st)
        try:
            text = Reply(**call.arguments).text
        except Exception as e:
            log.warning("[case] 回应格式不对（%s: %s），退回兜底台词", type(e).__name__, e)
            return self._fallback_reply(npc, st)
        ok, flags = self._check_output(text)
        if not ok:
            self.stats["guard_blocked"] += 1
            log.warning("[case][%s] output blocked %s: %s", npc.id, flags, text)
            return self._fallback_reply(npc, st)
        return text

    def _build_prompt(self, npc: CaseNPC, st: _NpcState) -> str:
        parts = [
            f"你是案件游戏里的 NPC「{npc.name}」（{npc.role}）。{npc.persona}",
            f"案发地点：{self.case.room}",
            "每次只说一句话，不超过 40 个字，要符合你的性格；不要用旁白或者动作描写，只说台词本身。",
            "玩家是被请来查这件事的调查员，你在被问话，不是在自由聊天，回答要跟问题相关。",
            "以下是你身上的秘密，每一条现在是「坦白」还是「隐瞒」状态已经由系统判定好了，你只需要照着说：",
        ]
        for s in npc.secrets:
            if s.fact_id in st.revealed:
                parts.append(f"- 关于「{s.summary}」：条件已经成立，这件事你现在必须老实说：{s.truth}")
            else:
                parts.append(
                    f"- 关于「{s.summary}」：还没到坦白的时候，如果被问起，用这个说法搪塞：{s.cover_story}"
                    "，口吻要自然、像是发自内心地这么认为，不要表现得心虚或者一提就否认得很用力。"
                )
        parts.append(
            "如果玩家问的事跟上面列的秘密都不相关，就正常地、符合你性格地回应，不用刻意回避。"
            "不要主动承认自己在说谎、不要提到「坦白条件」「系统判定」这些词——这是只有你知道的内心状态，"
            "不是台词的一部分。"
        )
        return "\n".join(parts)

    def _check_output(self, text: str) -> tuple[bool, list[str]]:
        """案件模式的出口检查：沿用小镇那套"出戏/泄露提示词"检测和长度上限，
        但不用 ungrounded_items——那是给小镇的食物/物品设定配的词表，案件是完全不同的世界观，
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
        hidden = [s for s in npc.secrets if s.fact_id not in st.revealed]
        return hidden[0].cover_story if hidden else "……"

    async def _call_tool(self, system: str, user: str, tool: dict, stat_key: str):
        self.stats[stat_key] += 1
        try:
            return await self.llm.choose_tool(system, user, [tool])
        except Exception as e:
            log.warning("[case] LLM 调用异常（%s: %s）", type(e).__name__, e)
            return None

    def resolve(self) -> Ending:
        return resolve_ending(frozenset(self.discovered))
