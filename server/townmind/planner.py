"""日程（规划层）：NPC 没人搭话的时候按"今天该在哪、干什么"行动，而不是随机闲逛。

Generative Agents 论文里 NPC 有三大件——记忆、反思、规划。前两件项目里早就有了，这是补
第三件。没有它，Alice 会在半夜跑去铁匠铺、Bob 下午在酒馆晃，小镇只有"碰面"没有"生活"。

三个组件：
  - GameClock：真实秒 -> 游戏时刻。一天等于 DAY_SECONDS 秒真实时间（默认 600，25 秒一个
    游戏小时），只做这一件事。以后要把记忆的时效衰减也换成游戏时间，从这里取。
  - DailyPlan / PlanBlock：一天 4~8 段，每段"几点到几点、在哪、干什么"。地点必须是
    world.LOCATIONS 里真实存在的（tool schema 里是 Literal 枚举，模型编不出地方来）。
  - template_plan()：按"工作地点 + 时段"拼的兜底日程。大模型不可用、生成失败、offline 评测，
    都用它——所以规划层不依赖大模型也能跑，跟行为树兜底是同一个哲学。

不做的（有意的）：论文里把一天递归细化到小时、再到十分钟——一层够了；对话打断日程之后
"重新规划"——对话结束自然回到当前时段该在的地方，不需要重算。
"""
import os
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from . import world
from .personas import DEFAULT_PERSONA, PERSONAS

ENV_DAY_SECONDS = "TOWNMIND_DAY_SECONDS"
DAY_SECONDS = 600.0  # 一个游戏日 = 多少秒真实时间（演示尺度：10 分钟一天，25 秒一小时）

PLACE_NAMES = tuple(loc.name for loc in world.LOCATIONS)


class GameClock:
    """真实时间 -> 游戏时刻。epoch 是"第 0 天 0 点"对应的真实时间戳。"""

    def __init__(self, day_seconds: float | None = None, epoch: float = 0.0) -> None:
        env = os.getenv(ENV_DAY_SECONDS)
        self.day_seconds = float(day_seconds or (float(env) if env else DAY_SECONDS))
        self.epoch = epoch

    def day(self, now: float) -> int:
        return int((now - self.epoch) // self.day_seconds)

    def hour(self, now: float) -> float:
        """0.0 <= hour < 24.0，带小数。"""
        return ((now - self.epoch) % self.day_seconds) / self.day_seconds * 24.0

    def describe(self, now: float) -> str:
        h = self.hour(now)
        hh, mm = int(h), int((h - int(h)) * 60)
        part = "凌晨" if h < 6 else "上午" if h < 12 else "下午" if h < 18 else "晚上"
        return f"{part} {hh:02d}:{mm:02d}"


# ---- 日程的数据结构（同时也是给大模型的 tool schema）----
class PlanBlock(BaseModel):
    start: int = Field(ge=0, le=23, description="开始的小时（0-23）")
    end: int = Field(ge=1, le=24, description="结束的小时（1-24），要大于开始")
    place: Literal[PLACE_NAMES] = Field(description="这段时间在哪个地点")  # type: ignore[valid-type]
    activity: str = Field(min_length=1, max_length=30, description="在做什么，一句话，不超过 15 个字")


class DailyPlan(BaseModel):
    """一天的日程。校验：按开始时间排序后不重叠。不要求覆盖全天——没排到的时段按
    "待在工作地点"处理（见 Planner.where）。"""

    blocks: list[PlanBlock] = Field(min_length=1, max_length=8)

    @field_validator("blocks")
    @classmethod
    def _ordered_and_disjoint(cls, blocks: list[PlanBlock]) -> list[PlanBlock]:
        blocks = sorted(blocks, key=lambda b: b.start)
        for b in blocks:
            if b.end <= b.start:
                raise ValueError(f"时段结束要晚于开始：{b.start}-{b.end}")
        for a, b in zip(blocks, blocks[1:]):
            if a.end > b.start:
                raise ValueError(f"日程时段重叠：{a.start}-{a.end} 与 {b.start}-{b.end}")
        return blocks

    def block_at(self, hour: float) -> PlanBlock | None:
        for b in self.blocks:
            if b.start <= hour < b.end:
                return b
        return None

    def describe(self) -> str:
        return "；".join(f"{b.start}-{b.end} 点在{b.place}{b.activity}" for b in self.blocks)


WRITE_PLAN_TOOL = {
    "name": "write_plan",
    "description": "给这个 NPC 安排今天的日程：4 到 6 段，每段几点到几点、在哪、做什么",
    "parameters": DailyPlan.model_json_schema(),
}


def template_plan(npc_id: str) -> DailyPlan:
    """兜底日程：白天在工作地点，中午和傍晚去广场或酒馆，夜里回工作地点休息。
    没有工作地点的（货郎 Milo）白天在广场和杂货铺之间。不聪明，但像个正常的镇民。"""
    p = PERSONAS.get(npc_id, DEFAULT_PERSONA)
    home = p.get("home") or ""
    if home not in PLACE_NAMES:
        home = "广场"
    social = "酒馆" if "酒馆" in PLACE_NAMES else home
    # 午休和傍晚社交的时段按 NPC 错开一小时：十个人同一分钟涌向广场、同一分钟涌向酒馆，
    # 不像小镇像集合哨——offline 评测第一次跑就是这样，"附近平均几个人"反而比闲逛还高
    k = list(PERSONAS).index(npc_id) % 3 if npc_id in PERSONAS else 0
    lunch, evening = 11 + k, 17 + k
    return DailyPlan(blocks=[
        PlanBlock(start=0, end=7, place=home, activity="休息"),
        PlanBlock(start=7, end=lunch, place=home, activity="干活"),
        PlanBlock(start=lunch, end=lunch + 1, place="广场", activity="歇一会"),
        PlanBlock(start=lunch + 1, end=evening, place=home, activity="干活"),
        PlanBlock(start=evening, end=evening + 3, place=social, activity="聊天放松"),
        PlanBlock(start=evening + 3, end=24, place=home, activity="休息"),
    ])


def plan_prompt(npc_id: str) -> tuple[str, str]:
    """让大模型按人设写日程的提示词。地点名单写进去只是给它看，真正的约束在 tool schema 的枚举里。"""
    p = PERSONAS.get(npc_id, DEFAULT_PERSONA)
    home = f"你的工作地点是{p['home']}。" if p.get("home") else "你没有固定的工作地点。"
    system = (
        f"你是游戏小镇里的 NPC「{p['name']}」。{p.get('persona', '')}{home}"
        f"小镇里的地点只有：{'、'.join(PLACE_NAMES)}。place 只能从这几个名字里选，"
        "没有「家」这个地点——你休息就在自己的工作地点，不要写「家」「家里」「工作地点」这类词。"
        "请给自己安排今天的日程：4 到 6 段，覆盖一整天（0 点到 24 点），时段不能重叠。"
        "要符合你的身份和性格：干活的时间在工作地点，吃饭、歇息、社交可以去广场或酒馆，夜里休息。"
        "每段的 activity 用一句话写清在做什么，不超过 15 个字。"
    )
    return system, "请安排今天的日程。"


HOME_WORDS = ("家", "住处", "住所", "工作地点", "岗位", "店里", "铺里")


def coerce_places(raw: dict, npc_id: str) -> dict:
    """校验之前先把模型写的地点名归一化。真实跑过一次：10 份日程里 4 份把 place 写成
    「家」「家中」「家里」「待在工作地点」——schema 里明明是六个地点的枚举。教训是 tool
    calling 的枚举对模型只是提示，不是硬约束，校验得自己做。这里把"家"一类的词映射到
    该 NPC 的工作地点（没有工作地点的映射到广场），别的认不出来的段直接丢掉——没排到的
    时段本来就按工作地点算（见 Planner.where），比整份日程退回模板浪费一次调用强。"""
    home = PERSONAS.get(npc_id, DEFAULT_PERSONA).get("home") or "广场"
    blocks = []
    for b in raw.get("blocks", []) if isinstance(raw, dict) else []:
        if not isinstance(b, dict):
            continue
        place = str(b.get("place", "")).strip()
        if place not in PLACE_NAMES:
            exact = next((n for n in PLACE_NAMES if n in place), None)  # "去酒馆" -> 酒馆
            if exact:
                place = exact
            elif any(w in place for w in HOME_WORDS):
                place = home
            else:
                continue
        blocks.append({**b, "place": place})
    return {**raw, "blocks": blocks} if isinstance(raw, dict) else raw


class Planner:
    """每个 NPC 每个游戏日一份日程，懒生成、缓存。生成交给外面传进来的协程（agent 负责
    调大模型和记账），这里只管"该问的时候问一次、失败用模板、问过就不再问"。"""

    def __init__(self, clock: GameClock) -> None:
        self.clock = clock
        self._plans: dict[tuple[str, int], DailyPlan] = {}
        self._source: dict[tuple[str, int], str] = {}  # "llm" / "template"

    def get(self, npc_id: str, now: float) -> DailyPlan | None:
        return self._plans.get((npc_id, self.clock.day(now)))

    def set(self, npc_id: str, now: float, plan: DailyPlan, source: str) -> None:
        key = (npc_id, self.clock.day(now))
        self._plans[key] = plan
        self._source[key] = source

    def source(self, npc_id: str, now: float) -> str | None:
        return self._source.get((npc_id, self.clock.day(now)))

    def where(self, npc_id: str, now: float) -> tuple[str, str] | None:
        """此刻该在哪、干什么。没有日程返回 None；有日程但这个小时没排到，按工作地点算。"""
        plan = self.get(npc_id, now)
        if plan is None:
            return None
        b = plan.block_at(self.clock.hour(now))
        if b is not None:
            return b.place, b.activity
        home = PERSONAS.get(npc_id, DEFAULT_PERSONA).get("home") or "广场"
        return home, "忙自己的事"
