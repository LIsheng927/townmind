"""关系状态（好感度/信任度）：有界更新、随时间回归中性、落盘往返。

这个模块是纯 stdlib（不依赖 pydantic / 网络），所以这些测试全是确定性的，
没有任何"跑两次结果可能不一样"的地方。"""
from townmind.social import (
    RELATION_BOUND,
    Relationship,
    RelationshipBook,
    _saturating_add,
)

NOW = 1000.0


# ---------- 有界更新：越接近边界，同方向越推不动；往回拉不打折 ----------
def test_saturating_add_barely_moves_near_bound():
    # 已经 9 分（接近上限 10）时再来一次 +3，只剩 10% 的空间，实际只加 0.3
    assert abs(_saturating_add(9.0, 3.0, 10.0) - 9.3) < 1e-9


def test_saturating_add_pullback_is_not_discounted():
    # 但反方向（失望）是实打实的：9 分挨一次 -3 就是 6 分。
    # 这个不对称是故意的——好感很高时再多夸一句没什么增量，一次失望却会实实在在掉分
    assert abs(_saturating_add(9.0, -3.0, 10.0) - 6.0) < 1e-9


def test_saturating_add_never_exceeds_bound():
    assert -10.0 <= _saturating_add(9.9, 100.0, 10.0) <= 10.0
    assert -10.0 <= _saturating_add(-9.9, -100.0, 10.0) <= 10.0


def test_saturating_add_pullback_from_bound_not_amplified():
    # 从 -10 往回拉时 room 会算出 2.0，如果不夹到 1.0，+3 会被放大成 +6
    assert abs(_saturating_add(-10.0, 3.0, 10.0) - (-7.0)) < 1e-9


def test_saturating_add_zero_delta_is_noop():
    assert _saturating_add(4.0, 0.0, 10.0) == 4.0


# ---------- 没打过交道的人 ----------
def test_stranger_is_neutral_and_not_stored():
    b = RelationshipBook()
    assert b.get("bob", NOW) == Relationship()
    assert b.known() == []  # 只是查一下，不该凭空多出一段空关系
    assert b.describe("bob", NOW) is None  # 没印象就不占提示词的地方


# ---------- 基本更新 ----------
def test_apply_records_delta_reason_and_count():
    b = RelationshipBook()
    b.apply("bob", 3, 2, "聊得挺投机", NOW)
    r = b.get("bob", NOW)
    assert r.affinity > 0 and r.trust > 0
    assert r.interactions == 1
    assert r.notes == ("聊得挺投机",)
    assert "聊得挺投机" in b.describe("bob", NOW)


def test_notes_keep_only_the_latest_few():
    b = RelationshipBook()
    for i in range(6):
        b.apply("eve", 1, 0, f"理由{i}", NOW + i)
    notes = b.get("eve", NOW + 6).notes
    assert len(notes) == 3  # MAX_NOTES
    assert notes[-1] == "理由5"


# ---------- 随时间回归中性 ----------
def test_impression_decays_by_half_after_half_life():
    b = RelationshipBook(half_life=100.0)
    b.apply("carol", 8, 8, "", NOW)
    before = b.get("carol", NOW).affinity
    assert abs(b.get("carol", NOW + 100.0).affinity - before / 2) < 1e-9


def test_decay_is_computed_at_read_time_not_stored():
    """衰减是读的时候按时间算出来的，不是定时任务去刷新每一对关系——NPC 一多，
    那种刷新是纯粹的浪费。所以"读过未来的值"不该把存着的值改掉。"""
    b = RelationshipBook(half_life=100.0)
    b.apply("carol", 8, 8, "", NOW)
    before = b.get("carol", NOW).affinity
    b.get("carol", NOW + 10000.0)  # 读一次很久以后的值
    assert abs(b.get("carol", NOW).affinity - before) < 1e-9


def test_apply_decays_before_adding_delta():
    """先衰减再加 delta：很久没见、今天聊得不错，结果应该是从"淡下来的基线"往上走，
    而不是从三个月前那个旧值往上走。"""
    b = RelationshipBook(half_life=100.0)
    b.apply("dave", 8, 0, "", NOW)
    faded = b.get("dave", NOW + 300.0).affinity  # 衰减到 1/8
    b.apply("dave", 1, 0, "", NOW + 300.0)
    assert b.get("dave", NOW + 300.0).affinity < faded + 1.01


def test_clock_going_backwards_does_not_amplify():
    b = RelationshipBook(half_life=100.0)
    b.apply("dave", 8, 0, "", NOW)
    assert abs(b.get("dave", NOW - 5000.0).affinity - 8.0) < 1e-9


# ---------- 描述文字 ----------
def test_describe_uses_name_and_reflects_negative_impression():
    b = RelationshipBook()
    b.apply("frank", -4, -6, "他骗了我", NOW)
    line = b.describe("frank", NOW, name="老弗")
    assert "老弗" in line
    assert "不" in line  # 负面印象要用负面的词
    assert b.trust_of("frank", NOW) < -1.5


# ---------- 落盘往返 ----------
def test_roundtrip_preserves_values_and_notes():
    b = RelationshipBook()
    b.apply("frank", -4, -6, "他骗了我", NOW)
    restored = RelationshipBook.from_dict(b.to_dict())
    assert abs(restored.get("frank", NOW).trust - b.get("frank", NOW).trust) < 1e-9
    assert restored.get("frank", NOW).notes == ("他骗了我",)


def test_bad_row_is_skipped_without_losing_the_rest():
    """单条读坏了只跳过这一条，不该让整本关系册跟着丢——跟 MemoryStore.load 的取舍一致：
    关系数据是锦上添花，坏了不该影响 NPC 还能不能正常说话。"""
    book = RelationshipBook.from_dict(
        {"relationships": {"x": {"affinity": "坏"}, "y": {"affinity": 1.0, "trust": 2.0}}}
    )
    assert book.known() == ["y"]


def test_from_dict_tolerates_garbage():
    assert RelationshipBook.from_dict(None).known() == []
    assert RelationshipBook.from_dict({}).known() == []


def test_bound_is_respected_after_many_updates():
    """有界更新的意义：几十次对话之后关系值应该收敛在区间里，而不是单调漂移到边界外。"""
    b = RelationshipBook(half_life=1e12)  # 基本不衰减，纯看有界更新本身
    for i in range(50):
        b.apply("bob", 3, 3, "", NOW + i)
    r = b.get("bob", NOW + 50)
    assert 0 < r.affinity <= RELATION_BOUND
    assert 0 < r.trust <= RELATION_BOUND
