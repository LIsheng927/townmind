"""小镇的"世界设定"：地点、每个地点的介绍和事实、镇上的公共事实。

这是 NPC 聊天和行动的事实来源：NPC 只能聊这里写到的东西，避免大模型凭空编造。
想修改小镇，只改这个文件：加地点、改事实都行。

当前用最简单的方式挑选"和 NPC 此刻相关的设定"：人在哪个地点，就给哪个地点的信息，
再加上镇上的公共事实。设定变多之后，第二小步会换成向量检索（RAG）。
"""
import math
import random
from dataclasses import dataclass

from .policy import WORLD_HALF_SIZE

AT_RADIUS = 3.0  # 离地点中心多近算"在这个地点"


def _clamp(v: float) -> float:
    return max(-WORLD_HALF_SIZE, min(WORLD_HALF_SIZE, v))


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    x: float
    z: float
    kind: str  # building / plaza，Unity 据此决定画成什么
    description: str
    facts: tuple[str, ...] = ()

    def stand_point(self, rng=None) -> tuple[float, float]:
        """NPC 前往这个地点时站的位置（建筑是站在门前，不是走进墙里）。"""
        rng = rng or random
        if self.kind == "plaza":
            x, z = self.x + rng.uniform(-1.5, 1.5), self.z + rng.uniform(-1.5, 1.5)
        else:
            x, z = self.x + rng.uniform(-1.0, 1.0), self.z - 2.0 + rng.uniform(-0.3, 0.3)
        return round(_clamp(x), 2), round(_clamp(z), 2)


LOCATIONS: tuple[Location, ...] = (
    Location(
        id="bakery",
        name="面包店",
        x=-5.0,
        z=4.0,
        kind="bakery",
        description="一家飘着香味的小面包店，Alice 在这里工作，卖法棍、肉桂卷和蓝莓松饼。",
        facts=(
            "面包店早上最热闹，法棍通常一出炉就卖光。",
            "最近面粉涨价了，Alice 有点发愁。",
        ),
    ),
    Location(
        id="smithy",
        name="铁匠铺",
        x=5.0,
        z=4.0,
        kind="smithy",
        description="叮叮当当的铁匠铺，Bob 在这里打铁，修农具、打刀具、钉马掌。",
        facts=("铁匠铺最近铁矿存货不多，Bob 正在等新的矿石送到。",),
    ),
    Location(
        id="plaza",
        name="广场",
        x=0.0,
        z=-4.0,
        kind="plaza",
        description="小镇中心的广场，有一口古井和几张长椅，旅行商人常在这里摆摊。",
        facts=(
            "广场上周六有集市，镇上的人都会来。",
            "广场的古井据说三十年没有干过。",
        ),
    ),
)

TOWN_FACTS: tuple[str, ...] = (
    "小镇很小，主要就是面包店、铁匠铺和广场这三个地方。",
    "旅行商人 Carol 刚到小镇不久，还不太熟悉这里。",
)

_BY_NAME = {loc.name: loc for loc in LOCATIONS}


def get_location(name: str) -> Location | None:
    return _BY_NAME.get(name)


def nearest_location(pos: tuple[float, float]) -> tuple[Location, float]:
    loc = min(LOCATIONS, key=lambda l: math.dist(pos, (l.x, l.z)))
    return loc, math.dist(pos, (loc.x, loc.z))


def location_at(pos: tuple[float, float], radius: float = AT_RADIUS) -> Location | None:
    loc, d = nearest_location(pos)
    return loc if d <= radius else None


def describe_surroundings(pos: tuple[float, float]) -> list[str]:
    """给提示词用：NPC 此刻所处位置的相关设定 + 镇上的公共事实。"""
    lines: list[str] = []
    loc = location_at(pos)
    if loc is not None:
        lines.append(f"你现在在：{loc.name}。{loc.description}")
        if loc.facts:
            lines.append("这里的情况：")
            lines += [f"- {f}" for f in loc.facts]
    else:
        near, d = nearest_location(pos)
        lines.append(f"你现在在小镇的空地上，离{near.name}最近（约 {d:.0f} 米）。")
    if TOWN_FACTS:
        lines.append("镇上的事：")
        lines += [f"- {f}" for f in TOWN_FACTS]
    return lines


def locations_payload() -> list[dict]:
    """发给 Unity 的地点列表（Unity 据此在场景里盖房子）。"""
    return [{"id": l.id, "name": l.name, "x": l.x, "z": l.z, "kind": l.kind} for l in LOCATIONS]
