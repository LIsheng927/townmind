"""小镇的"世界设定"：地点、每个地点的介绍和事实、镇上的公共事实。

这是 NPC 聊天和行动的事实来源：NPC 只能聊这里写到的东西，避免大模型凭空编造。
想修改小镇，只改这个文件：加地点、改事实都行。

当前所在地点的信息（描述 + 这个地点自己的 facts）始终全给——人已经站在那儿了，跟"此刻在聊
什么"无关，条数也一直很少，不用筛。"镇上的事"（TOWN_FACTS）不一样：不跟地点绑定、会持续
变多，所以由 agent.py 在有 embedder、且这一轮有 query（听到话或附近有人）时，跟记忆用同一套
语义检索先筛出最相关的几条，再传进 describe_surroundings；没有 query 或没配 embedder 时，
这里退回最初的做法——全部塞进去。
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
    Location(
        id="mill",
        name="磨坊",
        x=-7.0,
        z=-2.0,
        kind="building",
        description="镇西头的水磨坊，Elsa 在这里把麦子磨成面粉，面包店的面粉都是从这儿来的。",
        facts=(
            "磨坊的面粉都是供给面包店的，今年麦子收成一般，所以面粉才涨了价。",
            "磨坊的水轮去年修过一次，现在转起来还有点响。",
        ),
    ),
    Location(
        id="tavern",
        name="酒馆",
        x=7.0,
        z=-2.0,
        kind="building",
        description="镇东头的酒馆，Dan 在这里招呼客人，供应自家酿的麦酒、啤酒和一锅炖肉。",
        facts=(
            "酒馆是镇上消息最灵通的地方，谁家有什么事，在这儿坐一晚上多半能听着。",
            "酒馆晚上最热闹，白天基本没什么人。",
        ),
    ),
    Location(
        id="general_store",
        name="杂货铺",
        x=0.0,
        z=7.0,
        kind="building",
        description="镇北的杂货铺，Finn 在这里卖针线、盐巴、蜡烛、陶罐这些过日子的零碎东西。",
        facts=(
            "杂货铺什么都卖一点，但每样都不多，卖完了要等下次进货。",
            "镇上的人买不到的东西，一般会托货郎 Milo 从邻镇捎。",
        ),
    ),
)

TOWN_FACTS: tuple[str, ...] = (
    "小镇不大，一共就面包店、铁匠铺、磨坊、酒馆、杂货铺和中心广场这几个地方。",
    "镇上住着十来口人：面包师 Alice 和她的帮工 Iris、铁匠 Bob 和学徒 Greta、"
    "磨坊主 Elsa、酒馆老板 Dan、杂货铺老板 Finn、守夜人 Jonas，"
    "还有旅行商人 Carol 和走街串巷的货郎 Milo。",
    "旅行商人 Carol 刚到小镇不久，还不太熟悉这里。",
    "镇长每个月都会来广场巡视一圈，顺便查看一下治安。",
    "小镇建立差不多有五十年了，最早只有铁匠铺一家店，后来才慢慢热闹起来。",
    "小镇治安一向很好，镇上几乎没出过什么偷盗案件。",
    "小镇没有自己的医馆，谁生病了都要去邻镇看大夫。",
)


@dataclass(frozen=True)
class Item:
    """伙伴 NPC 能帮玩家搬运的东西。只记出生位置——捡起来之后东西在哪，
    由 Agent 动态维护（跟着拿它的人走，放下了就停在放下的地方），不属于"世界设定"这种静态事实。"""

    id: str
    name: str
    x: float
    z: float
    description: str


ITEMS: tuple[Item, ...] = (
    # 铁剑/银剑故意起了容易混淆的名字：玩家只说"剑"的时候，伙伴该反问是哪一把，而不是瞎猜
    Item(id="sword_iron", name="铁剑", x=6.0, z=3.0, description="铁匠铺门口刚打好的一把铁剑。"),
    Item(id="sword_silver", name="银剑", x=6.5, z=3.0, description="铁匠铺门口另一把镶银的剑，是镇长定做的。"),
    Item(id="bread_basket", name="面包篮", x=-6.0, z=3.0, description="面包店门口装面包用的藤条篮子。"),
)

_BY_NAME = {loc.name: loc for loc in LOCATIONS}
_ITEMS_BY_NAME = {i.name: i for i in ITEMS}


def get_location(name: str) -> Location | None:
    return _BY_NAME.get(name)


def get_item(name: str) -> Item | None:
    return _ITEMS_BY_NAME.get(name)


def is_near(a: tuple[float, float], b: tuple[float, float], radius: float = AT_RADIUS) -> bool:
    """通用的"离得够近"判断，捡/放东西的前置条件检查用得到，跟地点判断共用同一个半径。"""
    return math.dist(a, b) <= radius


def nearest_location(pos: tuple[float, float]) -> tuple[Location, float]:
    loc = min(LOCATIONS, key=lambda l: math.dist(pos, (l.x, l.z)))
    return loc, math.dist(pos, (loc.x, loc.z))


def location_at(pos: tuple[float, float], radius: float = AT_RADIUS) -> Location | None:
    loc, d = nearest_location(pos)
    return loc if d <= radius else None


def describe_surroundings(pos: tuple[float, float], town_facts: tuple[str, ...] | None = None) -> list[str]:
    """给提示词用：NPC 此刻所处位置的相关设定 + 镇上的事。

    town_facts：调用方（agent.py）语义检索之后筛出来的子集；传 None 时退回旧行为——
    TOWN_FACTS 全部塞进去。不传空元组和传 None 是两回事：空元组就是"筛完一条都不相关"，
    照样什么都不加；None 才是"没筛，别管我，全给"。
    """
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
    facts = TOWN_FACTS if town_facts is None else town_facts
    if facts:
        lines.append("镇上的事：")
        lines += [f"- {f}" for f in facts]
    return lines


def locations_payload() -> list[dict]:
    """发给 Unity 的地点列表（Unity 据此在场景里盖房子）。"""
    return [{"id": l.id, "name": l.name, "x": l.x, "z": l.z, "kind": l.kind} for l in LOCATIONS]


def items_payload() -> list[dict]:
    """发给 Unity 的物品列表（Unity 据此在场景里摆放可搬运的物件）。"""
    return [{"id": i.id, "name": i.name, "x": i.x, "z": i.z} for i in ITEMS]
