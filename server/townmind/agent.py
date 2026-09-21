"""NPC 的大脑：感知 -> 让 LLM 通过工具调用做决策 -> 校验 -> 返回动作。
任何环节失败（无 LLM、超时、报错、参数非法）都退回规则策略，保证 NPC 永远有动作可做。

NPC 之间的对话：某个 NPC 说话时，服务端把这句话记成一条"说话事件"（谁、在哪、说了什么）。
其他 NPC 下次决策时，如果当时就在附近，这句话会被写进它的提示词，它就"听到"了，
大模型据此决定是否回应。整个对话由大模型逐句生成，没有任何预设台词。"""
import asyncio
import logging
import math
import random
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from . import policy, world
from .llm.base import LLMClient, ToolCall
from .memory import Memory, MemoryStore, format_age
from .personas import DEFAULT_PERSONA, PERSONAS

log = logging.getLogger("townmind.agent")
HALF = policy.WORLD_HALF_SIZE
NEARBY_RADIUS = 5.0  # 这个距离内算"附近"，也是能听到说话的距离
EVENT_TTL = 30.0  # 说话事件的有效期（秒），太久以前的话不再被听到
SAY_COOLDOWN = 6.0  # 同一个 NPC 两次说话的最短间隔（秒）
CHAT_WINDOW = 30.0  # 统计"最近说了几句"的时间窗口（秒）
DISENGAGE_SECONDS = 20.0  # 道别之后这么久内不再和人搭话，走去忙自己的事
# 记忆的重要度（1-10）：第一版用简单规则打分
IMPORTANCE_MET = 8  # 第一次见到某人
IMPORTANCE_HEARD = 6  # 别人对我说的话
IMPORTANCE_FAREWELL = 5
IMPORTANCE_SAID = 4  # 我自己说的话
# 走路和休息不值得记：占位置、没信息量，所以不存
MAX_SAYS_PER_WINDOW = 3  # 窗口内最多说几句，说满就该走开去忙别的，避免无限聊天烧钱


PLACE_NAMES = tuple(loc.name for loc in world.LOCATIONS)


class GoTo(BaseModel):
    place: Literal[PLACE_NAMES] = Field(description="要前往的地点名称")  # 只能是小镇里真实存在的地点


class Say(BaseModel):
    text: str = Field(min_length=1, max_length=60, description="要说的话，不超过 30 个字")


class Idle(BaseModel):
    seconds: float = Field(default=3, ge=0, le=10, description="原地停留的秒数")


class EndConversation(BaseModel):
    farewell: str = Field(min_length=1, max_length=60, description="道别的话，不超过 30 个字")


ARG_MODELS: dict[str, type[BaseModel]] = {
    "go_to": GoTo,
    "say": Say,
    "idle": Idle,
    "end_conversation": EndConversation,
}
TOOL_DESCRIPTIONS = {
    "go_to": f"前往小镇里的一个地点，可选：{'、'.join(PLACE_NAMES)}",
    "say": "说一句话（头顶会显示对话气泡，附近的人能听到）",
    "idle": "原地休息一会儿",
    "end_conversation": "结束当前的对话：说一句道别的话，然后走开去忙自己的事",
}
TOOLS = [
    {"name": n, "description": TOOL_DESCRIPTIONS[n], "parameters": m.model_json_schema()}
    for n, m in ARG_MODELS.items()
]


@dataclass
class SpeechEvent:
    id: int
    speaker: str
    pos: tuple[float, float]  # 说话时说话者所在的位置
    text: str
    time: float


def _name(npc_id: str) -> str:
    return PERSONAS.get(npc_id, DEFAULT_PERSONA)["name"]


class Agent:
    def __init__(
        self,
        llm: LLMClient | None,
        timeout: float = 8.0,
        clock: Callable[[], float] = time.time,  # 用真实时间：记忆要能跨重启，monotonic 重启后会归零
        memory_dir: Path | None = None,
        rng: random.Random | None = None,
        use_memory: bool = True,  # 评测时可关闭，做消融对比
        use_lore: bool = True,
    ) -> None:
        self.llm = llm
        self.rng = rng or random.Random()
        self.use_memory = use_memory
        self.use_lore = use_lore
        self.trace: list[dict] | None = None  # 不为 None 时，每次决策都记一笔，供评测使用
        self.timeout = timeout
        self.clock = clock  # 可注入，测试时用假时钟
        self.positions: dict[str, tuple[float, float]] = {}
        self.memory_dir = memory_dir  # 为 None 时记忆只存在内存里（测试用）
        self._memories: dict[str, MemoryStore] = {}
        self.events: deque[SpeechEvent] = deque(maxlen=50)
        self._next_event_id = 1
        self.last_heard: dict[str, int] = defaultdict(int)  # 每个 NPC 已经处理到的事件 id
        self.last_said: dict[str, float] = {}
        self.say_times: dict[str, deque[float]] = defaultdict(deque)
        self.disengaged_until: dict[str, float] = {}  # 道别后，在这个时间点之前不再搭话
        # 决策统计：用来观察成本，也是后面评测框架的基础
        self.stats: dict[str, int] = defaultdict(int)

    def update_position(self, npc_id: str, pos) -> None:
        """记录 NPC 的最新位置。Unity 走路时会定期上报，所以"谁在附近"不会用过期位置来判断。"""
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            self.positions[npc_id] = (float(pos[0]), float(pos[1]))

    async def decide(self, npc_id: str, observation: dict) -> dict:
        self.update_position(npc_id, observation.get("pos"))

        now = self.clock()
        heard = self._collect_heard(npc_id, now)  # 先取走"听到的话"
        nearby = self._nearby(npc_id)
        interesting = bool(heard or nearby)  # 有人在附近或刚听到话，才算"有事发生"
        status = self._say_status(npc_id, now)
        # 回忆：只取和眼前的人最相关、最重要、最新的几条。要在写入本轮新记忆之前取，避免"想起"刚发生的事
        involved = {o for o, _ in nearby} | {e.speaker for e in heard}
        recalled = self._mem(npc_id).recall(involved, now) if self.use_memory else []

        action, source = None, "fallback"
        if self.llm is not None:
            if interesting and status == "ok":
                # 只有这一种情况才花钱问大模型
                action, source = await self._ask_llm(npc_id, heard, nearby, recalled, now)
            elif interesting and status == "cooldown":
                # 刚说过话，还不能再说：原地等对方回应，而不是走开
                wait = SAY_COOLDOWN - (now - self.last_said[npc_id])
                action, source = {"name": "idle", "seconds": round(max(1.0, min(wait, 10.0)), 1)}, "rule"
            else:
                # 没事发生，或者这场对话已经聊够了：走走停停，不调用大模型
                action, source = self._wander(npc_id), "rule"

        if action is None:
            action, source = policy.decide(npc_id, observation), "fallback"

        if action["name"] == "go_to":
            # 大模型只选"去哪个地点"；坐标由服务端根据世界设定算出来。Unity 仍然只认 move_to。
            loc = world.get_location(action["place"])
            x, z = loc.stand_point(self.rng)
            action = {"name": "move_to", "x": x, "z": z, "place": loc.name}

        ended = action["name"] == "end_conversation"
        if ended:
            # Unity 只认识 say/move_to/idle，所以这里把它翻译成"说一句道别"，
            # 同时让服务端记住：这个 NPC 接下来一阵子要走开，不再问大模型。
            self.stats["ended_conversations"] += 1
            self.disengaged_until[npc_id] = now + DISENGAGE_SECONDS
            action = {"name": "say", "text": action["farewell"]}

        self.stats[source] += 1
        if action["name"] == "say":
            self._record_speech(npc_id, action["text"], now)

        if self.use_memory:
            self._remember(npc_id, now, heard, nearby, action, ended)
        if self.trace is not None:
            self.trace.append(
                {"t": now, "npc": npc_id, "source": source, "action": dict(action), "nearby": [o for o, _ in nearby]}
            )
        log.info("[%s] %s -> %s%s", npc_id, source, action, "  (end_conversation)" if ended else "")
        return action

    async def _ask_llm(self, npc_id, heard, nearby, recalled, now):
        system, user = self._build_prompt(npc_id, heard, nearby, recalled, now)
        self.stats["llm_calls"] += 1
        try:
            call = await asyncio.wait_for(self.llm.choose_tool(system, user, TOOLS), self.timeout)
            self.stats["tokens_in"] += call.input_tokens  # 先记账再校验：校验失败的调用也是花了钱的
            self.stats["tokens_out"] += call.output_tokens
            return self._validate(call), "llm"
        except Exception as e:  # 超时/网络/参数非法都走兜底
            self.stats["llm_failures"] += 1
            log.warning("[%s] LLM failed (%s: %s), using fallback", npc_id, type(e).__name__, e)
            return None, "fallback"

    def _wander(self, npc_id: str) -> dict:
        """没事发生时的日常：一部分时间发呆，其余时间在小镇的真实地点之间走动，偏爱自己的工作地点。零成本。"""
        if self.rng.random() < 0.35:
            return {"name": "idle", "seconds": round(self.rng.uniform(2, 5), 1)}
        home = world.get_location(PERSONAS.get(npc_id, DEFAULT_PERSONA).get("home", ""))
        loc = home if (home and self.rng.random() < 0.5) else self.rng.choice(world.LOCATIONS)
        x, z = loc.stand_point(self.rng)
        return {"name": "move_to", "x": x, "z": z, "place": loc.name}

    # ---------- 记忆 ----------
    def _memory_path(self, npc_id: str) -> Path | None:
        if self.memory_dir is None:
            return None
        return self.memory_dir / (re.sub(r"[^A-Za-z0-9_-]", "_", npc_id) + ".json")

    def _mem(self, npc_id: str) -> MemoryStore:
        store = self._memories.get(npc_id)
        if store is None:
            path = self._memory_path(npc_id)
            store = MemoryStore.load(path) if path else MemoryStore()
            self._memories[npc_id] = store
        return store

    def _remember(self, npc_id, now, heard, nearby, action, ended) -> None:
        """把这一轮发生的事写进记忆，并给每条打上重要度。"""
        store = self._mem(npc_id)
        for e in heard:
            store.add(f"{_name(e.speaker)}对你说：「{e.text}」", IMPORTANCE_HEARD, now, {e.speaker})
        for o, _ in nearby:
            if o not in store.met:
                store.met.add(o)
                store.add(f"你第一次见到{_name(o)}", IMPORTANCE_MET, now, {o})
        near_ids = {o for o, _ in nearby}
        names = "、".join(_name(o) for o in sorted(near_ids))
        if action["name"] == "say":
            to = f"对{names}" if names else ""
            if ended:
                store.add(f"你{to}道别：「{action['text']}」", IMPORTANCE_FAREWELL, now, near_ids)
            else:
                store.add(f"你{to}说了「{action['text']}」", IMPORTANCE_SAID, now, near_ids)
        path = self._memory_path(npc_id)
        if path is not None:
            try:
                store.save(path)
            except OSError as e:  # 存盘失败不应该影响游戏
                log.warning("[%s] failed to save memory: %s", npc_id, e)

    def memory_dump(self, npc_id: str) -> list[dict]:
        return self._mem(npc_id).dump(self.clock())

    def _say_status(self, npc_id: str, now: float) -> str:
        """ok：可以说话；cooldown：刚说过，稍等；capped：这一阵说得够多了，该走开；
        away：已经主动道别，正在走开。"""
        if now < self.disengaged_until.get(npc_id, 0.0):
            return "away"
        times = self.say_times[npc_id]
        while times and now - times[0] > CHAT_WINDOW:
            times.popleft()
        if len(times) >= MAX_SAYS_PER_WINDOW:
            return "capped"
        last = self.last_said.get(npc_id)
        if last is not None and now - last < SAY_COOLDOWN:
            return "cooldown"
        return "ok"

    def _nearby(self, npc_id: str) -> list[tuple[str, float]]:
        me = self.positions.get(npc_id)
        if me is None:
            return []
        return [
            (o, math.dist(me, pos))
            for o, pos in self.positions.items()
            if o != npc_id and math.dist(me, pos) <= NEARBY_RADIUS
        ]

    def _record_speech(self, npc_id: str, text: str, now: float) -> None:
        pos = self.positions.get(npc_id, (0.0, 0.0))
        self.events.append(SpeechEvent(self._next_event_id, npc_id, pos, text, now))
        self._next_event_id += 1
        self.last_said[npc_id] = now
        self.say_times[npc_id].append(now)

    def _collect_heard(self, npc_id: str, now: float) -> list[SpeechEvent]:
        me = self.positions.get(npc_id)
        heard: list[SpeechEvent] = []
        if me is not None:
            for e in self.events:
                if (
                    e.speaker != npc_id
                    and e.id > self.last_heard[npc_id]
                    and now - e.time <= EVENT_TTL
                    and math.dist(me, e.pos) <= NEARBY_RADIUS
                ):
                    heard.append(e)
        # 无论听没听到，都把指针推到最新：走远的人事后不会"补听"以前的话
        self.last_heard[npc_id] = self._next_event_id - 1
        return heard

    @staticmethod
    def _validate(call: ToolCall) -> dict:
        model = ARG_MODELS.get(call.name)
        if model is None:
            raise ValueError(f"unknown tool {call.name!r}")
        return {"name": call.name, **model(**call.arguments).model_dump()}

    def _build_prompt(self, npc_id, heard, nearby, recalled: list[Memory], now: float) -> tuple[str, str]:
        p = PERSONAS.get(npc_id, DEFAULT_PERSONA)
        parts = [
            f"你是游戏小镇里的 NPC「{p['name']}」。{p['persona']}",
            f"你的工作地点是{p['home']}。" if p.get("home") else "",
            f"小镇里有这些地点：{'、'.join(PLACE_NAMES)}。想去某处时用 go_to。",
            "每次你必须调用一个工具来决定下一步行动。",
            "你只能聊小镇里真实存在的事，也就是下面\"你现在在\"和\"镇上的事\"里写到的内容，不要编造不存在的地点、活动或人物。如果别人提到的地点、活动或人物不在设定里，就坦率说你没听说过，不要附和、也不要猜测。"
            if self.use_lore
            else "",
            f"只有附近（{NEARBY_RADIUS} 米内）有其他人时才说话，台词不超过 30 个字，要符合你的性格。",
            "如果刚有人对你说话，应当用 say 回应，形成一来一回的对话。",
            "话题聊完了、或者已经聊了三四句，就用 end_conversation 道别并走开去忙自己的事，不要一直聊下去。",
        ]
        system = "\n".join(x for x in parts if x)
        me = self.positions.get(npc_id, (0.0, 0.0))
        nearby_text = [f"{_name(o)}（距离 {d:.1f} 米）" for o, d in nearby]
        surroundings = (
            world.describe_surroundings(me) if self.use_lore else [f"你现在的位置：({me[0]:.1f}, {me[1]:.1f})"]
        )
        lines = surroundings + [
            f"附近的人：{'、'.join(nearby_text) if nearby_text else '没有人'}",
        ]
        if recalled:
            lines.append("你想起了：")
            lines += [f"- {format_age(now - m.time)}：{m.text}" for m in recalled]
        said = len(self.say_times[npc_id])
        if said:
            lines.append(f"你在最近 {CHAT_WINDOW:.0f} 秒内已经说了 {said} 句话（最多 {MAX_SAYS_PER_WINDOW} 句）。")
        if heard:
            lines.append("你刚听到：" + "；".join(f"{_name(e.speaker)}说「{e.text}」" for e in heard))
        lines.append("请决定下一步。")
        return system, "\n".join(lines)
