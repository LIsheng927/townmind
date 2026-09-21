"""大模型不可用时（没 key、超时、报错、参数非法）NPC 的行为树兜底：不聪明，但像个正常的 NPC。

优先级从高到低：
  1. 有人在附近且允许说话 -> 说一句这个角色的固定台词（听到了话就用"回应"类台词，否则用"打招呼"类）
  2. 不在自己的工作地点 -> 走回去
  3. 已经在岗位上 -> 大半时间歇着，偶尔去别的地点逛逛
"""
import random
from dataclasses import dataclass, field

from . import world
from .bt import Action, Condition, Selector, Sequence

DEFAULT_LINES = {"greet": ("你好。", "嗨。"), "reply": ("嗯，是这样。", "好的。")}


@dataclass
class Ctx:
    npc_id: str
    pos: tuple[float, float]
    home: str  # 工作地点名，可能为空
    lines: dict[str, tuple[str, ...]]
    nearby: list[str]  # 附近的人（显示名）
    heard: bool  # 刚刚是否听到有人说话
    can_say: bool  # 说话频率限制是否允许
    rng: random.Random
    action: dict | None = field(default=None)


def _say(ctx: Ctx) -> dict:
    key = "reply" if ctx.heard else "greet"
    return {"name": "say", "text": ctx.rng.choice(ctx.lines.get(key) or DEFAULT_LINES[key])}


def _move_to(loc: world.Location, ctx: Ctx) -> dict:
    x, z = loc.stand_point(ctx.rng)
    return {"name": "move_to", "x": x, "z": z, "place": loc.name}


def _at_home(ctx: Ctx) -> bool:
    here = world.location_at(ctx.pos)
    return bool(ctx.home) and here is not None and here.name == ctx.home


def _go_home(ctx: Ctx) -> dict | None:
    loc = world.get_location(ctx.home)
    return _move_to(loc, ctx) if loc else None


def _rest_or_visit(ctx: Ctx) -> dict:
    if ctx.rng.random() < 0.7:
        return {"name": "idle", "seconds": round(ctx.rng.uniform(2, 5), 1)}
    return _move_to(ctx.rng.choice(world.LOCATIONS), ctx)


def _wander(ctx: Ctx) -> dict:
    return _move_to(ctx.rng.choice(world.LOCATIONS), ctx)


TREE = Selector(
    Sequence(Condition(lambda c: c.can_say and (c.heard or bool(c.nearby))), Action(_say)),
    Sequence(Condition(lambda c: bool(c.home) and not _at_home(c)), Action(_go_home)),
    Sequence(Condition(_at_home), Action(_rest_or_visit)),
    Action(_wander),  # 没有工作地点的路人：随便逛
)


def decide(ctx: Ctx) -> dict:
    TREE.tick(ctx)
    assert ctx.action is not None  # 最后一个 Action 永远成功
    return ctx.action
