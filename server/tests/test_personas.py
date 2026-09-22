"""人设和世界设定的一致性检查。

这些断言单独看都很琐碎，但它们守的是同一类 bug：人设和世界设定是两份手写的数据，
改了一边忘了另一边不会报错，只会让 NPC 在运行时说出奇怪的话——比如工作地点写了个
不存在的名字（NPC 会永远走不到"家"），或者人设里提到一样设定里没有的东西（出口检查
会把这句话当成编造拦下来，NPC 于是永远退化成行为树的固定台词）。这两种都不会抛异常，
只会让效果悄悄变差，所以值得用测试钉住。
"""
from collections import Counter

from townmind import safety, world
from townmind.personas import DEFAULT_PERSONA, PERSONAS

NPCS = [k for k in PERSONAS if k != "player"]


def persona_text(npc_id: str) -> str:
    p = PERSONAS[npc_id]
    return (
        p["persona"]
        + "".join(p.get("speech_habits", ()))
        + "".join(line for group in p.get("lines", {}).values() for line in group)
    )


def test_every_npc_has_a_name_and_persona():
    for npc_id in NPCS:
        p = PERSONAS[npc_id]
        assert p.get("name"), npc_id
        assert p.get("persona"), npc_id


def test_every_npc_home_is_a_real_location():
    """工作地点写错名字不会报错，只会让 NPC 永远走不回"家"——行为树里
    "不在自己的工作地点就走回去"这条分支会一直触发不了。"""
    for npc_id in NPCS:
        home = PERSONAS[npc_id]["home"]
        if home:  # 空字符串是合法的：货郎 Milo 就没有固定摊位
            assert world.get_location(home) is not None, f"{npc_id} 的工作地点 {home} 不存在"


def test_every_npc_has_fallback_lines_for_both_situations():
    """大模型不可用时行为树要用这些固定台词。缺了的话所有人退化成同一套默认台词，
    一整镇的人说话一个味儿。"""
    for npc_id in NPCS:
        lines = PERSONAS[npc_id].get("lines", {})
        for key in ("greet", "reply"):
            assert lines.get(key), f"{npc_id} 缺 {key} 台词"


def test_every_npc_has_speech_habits():
    """expressive_dialogue 打开时靠这个拉开说话风格的差异。"""
    for npc_id in NPCS:
        assert PERSONAS[npc_id].get("speech_habits"), npc_id


def test_persona_text_does_not_trip_the_ungrounded_item_guard():
    """人设和固定台词里提到的东西，必须是设定里有的。

    否则出口检查会把 NPC 自己的固定台词当成"编造物品"拦下来——Dan 说一句
    「来碗啤酒」就被拦，是这个项目真实踩过的那类坑（啤酒里的"啤"+"酒"正好命中
    safety._ITEM_CORE 的食物词规则，必须显式加进 ALLOWED_ITEMS）。"""
    for npc_id in NPCS:
        assert safety.ungrounded_items(persona_text(npc_id)) == [], npc_id


def test_world_lore_does_not_trip_the_ungrounded_item_guard():
    for loc in world.LOCATIONS:
        assert safety.ungrounded_items(loc.description + "".join(loc.facts)) == [], loc.name
    for fact in world.TOWN_FACTS:
        assert safety.ungrounded_items(fact) == [], fact


def test_npc_names_are_unique():
    """重名的话，提示词里"Bob 对你说"会指向两个人，记忆里的 people 也分不清谁是谁。"""
    names = [PERSONAS[n]["name"] for n in NPCS]
    assert len(set(names)) == len(names), Counter(names).most_common(3)


def test_player_is_not_treated_as_an_npc():
    assert PERSONAS["player"]["name"] == "玩家"
    assert not PERSONAS["player"]["home"]


def test_default_persona_is_usable_for_unknown_ids():
    """没登记过的 npc_id 也得能正常跑（_name / _build_prompt 都会走 PERSONAS.get(..., DEFAULT)）。"""
    assert DEFAULT_PERSONA["name"] and DEFAULT_PERSONA["persona"]
    assert PERSONAS.get("不存在的人", DEFAULT_PERSONA)["name"] == DEFAULT_PERSONA["name"]


def test_most_workplaces_are_shared_so_npcs_actually_meet():
    """有人搭伙才会天天碰面、自然产生 NPC 之间的对话，不用全靠玩家去撩。
    这条钉住的是设计意图：至少有几个地点住着不止一个人。"""
    homes = Counter(PERSONAS[n]["home"] for n in NPCS if PERSONAS[n]["home"])
    shared = [place for place, count in homes.items() if count >= 2]
    assert len(shared) >= 2, homes


def test_sim_start_positions_cover_every_npc():
    """评测用的仿真出生点要和人设表对得上：漏了谁，那个 NPC 在整个评测里根本不会出场，
    评测结果会悄悄少算一个人。"""
    from evals.sim import START_POSITIONS

    assert set(START_POSITIONS) == set(NPCS)


def test_sim_start_positions_are_inside_the_world():
    from evals.sim import START_POSITIONS

    for npc_id, (x, z) in START_POSITIONS.items():
        assert abs(x) <= 8 and abs(z) <= 8, npc_id
