"""NPC 的大脑：感知 -> 让 LLM 通过工具调用做决策 -> 校验 -> 返回动作。
任何环节失败（无 LLM、超时、报错、参数非法）都退回规则策略，保证 NPC 永远有动作可做。"""
import asyncio
import logging
import math
from collections import defaultdict, deque

from pydantic import BaseModel, Field

from . import policy
from .llm.base import LLMClient, ToolCall
from .personas import DEFAULT_PERSONA, PERSONAS

log = logging.getLogger("townmind.agent")
HALF = policy.WORLD_HALF_SIZE
NEARBY_RADIUS = 5.0


class MoveTo(BaseModel):
    x: float = Field(ge=-HALF, le=HALF, description="目标 x 坐标")
    z: float = Field(ge=-HALF, le=HALF, description="目标 z 坐标")


class Say(BaseModel):
    text: str = Field(min_length=1, max_length=60, description="要说的话，不超过 30 个字")


class Idle(BaseModel):
    seconds: float = Field(default=3, ge=0, le=10, description="原地停留的秒数")


ARG_MODELS: dict[str, type[BaseModel]] = {"move_to": MoveTo, "say": Say, "idle": Idle}
TOOL_DESCRIPTIONS = {
    "move_to": "走到小镇里的某个位置",
    "say": "说一句话（头顶会显示对话气泡）",
    "idle": "原地休息一会儿",
}
TOOLS = [
    {"name": n, "description": TOOL_DESCRIPTIONS[n], "parameters": m.model_json_schema()}
    for n, m in ARG_MODELS.items()
]


class Agent:
    def __init__(self, llm: LLMClient | None, timeout: float = 8.0) -> None:
        self.llm = llm
        self.timeout = timeout
        self.positions: dict[str, tuple[float, float]] = {}
        self.history: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=3))

    async def decide(self, npc_id: str, observation: dict) -> dict:
        pos = observation.get("pos")
        if isinstance(pos, (list, tuple)) and len(pos) == 2:
            self.positions[npc_id] = (float(pos[0]), float(pos[1]))

        action, source = None, "fallback"
        if self.llm is not None:
            system, user = self._build_prompt(npc_id)
            try:
                call = await asyncio.wait_for(self.llm.choose_tool(system, user, TOOLS), self.timeout)
                action = self._validate(call)
                source = "llm"
            except Exception as e:  # 超时/网络/参数非法都走兜底
                log.warning("[%s] LLM failed (%s: %s), using fallback", npc_id, type(e).__name__, e)

        if action is None:
            action = policy.decide(npc_id, observation)
        self.history[npc_id].append(self._describe(action))
        log.info("[%s] %s -> %s", npc_id, source, action)
        return action

    @staticmethod
    def _validate(call: ToolCall) -> dict:
        model = ARG_MODELS.get(call.name)
        if model is None:
            raise ValueError(f"unknown tool {call.name!r}")
        return {"name": call.name, **model(**call.arguments).model_dump()}

    @staticmethod
    def _describe(action: dict) -> str:
        if action["name"] == "move_to":
            return f"走到了 ({action['x']}, {action['z']})"
        if action["name"] == "say":
            return f"说了「{action['text']}」"
        return "休息了一会儿"

    def _build_prompt(self, npc_id: str) -> tuple[str, str]:
        p = PERSONAS.get(npc_id, DEFAULT_PERSONA)
        system = (
            f"你是游戏小镇里的 NPC「{p['name']}」。{p['persona']}\n"
            f"小镇是一块平地，坐标 x、z 的范围都是 -{HALF} 到 {HALF}。\n"
            "每次你必须调用一个工具来决定下一步行动。"
            f"只有附近（{NEARBY_RADIUS} 米内）有其他人时才说话，台词不超过 30 个字，要符合你的性格。"
        )
        me = self.positions.get(npc_id, (0.0, 0.0))
        nearby = [
            f"{PERSONAS.get(o, DEFAULT_PERSONA)['name']}（距离 {math.dist(me, pos):.1f} 米）"
            for o, pos in self.positions.items()
            if o != npc_id and math.dist(me, pos) <= NEARBY_RADIUS
        ]
        recent = "；".join(self.history[npc_id]) or "无"
        user = (
            f"你现在的位置：({me[0]:.1f}, {me[1]:.1f})\n"
            f"附近的人：{'、'.join(nearby) if nearby else '没有人'}\n"
            f"你最近做过的事：{recent}\n请决定下一步。"
        )
        return system, user
