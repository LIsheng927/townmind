import json

from townmind.memory import (
    HOP_IMPORTANCE_DECAY,
    SHARE_MIN_IMPORTANCE,
    MemoryStore,
    format_age,
    retold_importance,
)

NOW = 1000.0


def carol_store() -> MemoryStore:
    s = MemoryStore()
    s.add("Bob 说他很忙", 6, NOW - 300, {"bob"})
    s.add("你对 Bob 打了招呼", 4, NOW - 180, {"bob"})
    s.add("你走到了某处", 1, NOW - 10)
    return s


def test_recall_prefers_relevant_important_over_recent_trivia():
    got = [m.text for m in carol_store().recall({"bob"}, NOW, k=3)]
    assert got == ["Bob 说他很忙", "你对 Bob 打了招呼", "你走到了某处"]


def test_without_relevance_recent_trivia_wins():
    # 眼前没有 Bob 时，相关度加分消失，最新的琐事反而排第一
    assert carol_store().recall(set(), NOW, k=1)[0].text == "你走到了某处"


def test_recall_respects_k():
    assert len(carol_store().recall({"bob"}, NOW, k=2)) == 2


def test_recency_halves_every_half_life():
    s = MemoryStore(half_life=120)
    s.add("x", 5, NOW - 120)
    m = s.memories[0]
    assert abs(s.score(m, frozenset(), NOW) - (0.5 + 0.5)) < 1e-9  # 新近度 0.5 + 重要度 0.5


def test_importance_is_clamped():
    s = MemoryStore()
    s.add("a", 99, NOW)
    s.add("b", -5, NOW)
    assert [m.importance for m in s.memories] == [10, 1]


def test_capacity_evicts_old_unimportant_memory():
    s = MemoryStore(capacity=3)
    s.add("a", 9, NOW)
    s.add("b", 8, NOW)
    s.add("琐事", 1, NOW - 500)
    s.add("c", 7, NOW)
    assert len(s.memories) == 3
    assert "琐事" not in [m.text for m in s.memories]


def test_save_and_load_roundtrip(tmp_path):
    s = carol_store()
    s.met.add("bob")
    path = tmp_path / "carol.json"
    s.save(path)
    loaded = MemoryStore.load(path)
    assert [m.text for m in loaded.memories] == [m.text for m in s.memories]
    assert loaded.met == {"bob"}
    assert loaded.memories[0].people == frozenset({"bob"})


def test_load_missing_file_is_empty(tmp_path):
    assert MemoryStore.load(tmp_path / "nope.json").memories == []


def test_load_corrupted_file_is_empty_not_crash(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("这不是 json", encoding="utf-8")
    assert MemoryStore.load(path).memories == []


def test_format_age():
    assert format_age(3) == "刚才"
    assert format_age(30) == "30 秒前"
    assert format_age(180) == "3 分钟前"
    assert format_age(7300) == "2 小时前"


# ---------- 语义相关度：Stanford「Generative Agents」那套检索公式里的 relevance 项 ----------
# 没有 query_embedding 时（没配 embedder，或者这条记忆本身没算过向量），相关度退化成
# 上面那套"认不认人"的旧公式——上面所有测试完全不用改，就是在验证这条退化路径。
# 这里单独测"两边都有向量"时，relevance 真的按内容语义算，而不是按人。
def test_semantic_relevance_beats_people_overlap_when_content_matches():
    s = MemoryStore()
    # "剑" 和 "面包" 两个话题，用正交向量模拟"完全不像" / "完全一样"
    s.add("有人在讨论铁匠铺的剑", 5, NOW, {"bob"}, embedding=(1.0, 0.0))
    s.add("有人在讨论面包店的面包", 5, NOW, {"bob"}, embedding=(0.0, 1.0))
    # 查询是"剑"的话题，即使两条记忆涉及的人（bob）完全一样，语义相关的那条也该排前面
    got = s.recall({"bob"}, NOW, k=2, query_embedding=(1.0, 0.0))
    assert [m.text for m in got] == ["有人在讨论铁匠铺的剑", "有人在讨论面包店的面包"]


def test_semantic_relevance_falls_back_to_people_overlap_when_memory_has_no_embedding():
    """老记忆（加语义检索之前存的）没有 embedding，即使传了 query_embedding，
    这一条也不该报错，relevance 退回"认不认人"那一版。"""
    s = MemoryStore()
    s.add("没算过向量的老记忆", 5, NOW, {"bob"})  # embedding 默认 None
    m = s.memories[0]
    assert s.score(m, {"bob"}, NOW, query_embedding=(1.0, 0.0)) == s.score(m, {"bob"}, NOW)


def test_negative_cosine_similarity_does_not_go_below_zero_relevance():
    """完全相反方向的向量，相关度不该变成负数拖累总分——按 0 分算，跟旧版"没命中"一个量级。"""
    s = MemoryStore()
    s.add("反方向的话题", 5, NOW, embedding=(1.0, 0.0))
    m = s.memories[0]
    score = s.score(m, frozenset(), NOW, query_embedding=(-1.0, 0.0))
    assert abs(score - (1.0 + 0.5)) < 1e-9  # 新近度 1.0 + 重要度 0.5 + 相关度 0（不是 -1）


def test_save_and_load_roundtrip_preserves_embedding(tmp_path):
    s = MemoryStore()
    s.add("有向量的记忆", 5, NOW, {"bob"}, embedding=(0.1, 0.2, 0.3))
    path = tmp_path / "with_embedding.json"
    s.save(path)
    loaded = MemoryStore.load(path)
    assert loaded.memories[0].embedding == (0.1, 0.2, 0.3)


def test_load_old_file_without_embedding_field_defaults_to_none(tmp_path):
    """兼容加这个功能之前存的记忆文件：没有 "embedding" 这个字段，读出来该是 None，不该报错。"""
    import json

    path = tmp_path / "old_format.json"
    path.write_text(
        json.dumps({"met": [], "memories": [{"text": "老格式", "time": NOW, "importance": 5, "people": []}]}),
        encoding="utf-8",
    )
    loaded = MemoryStore.load(path)
    assert loaded.memories[0].embedding is None


# ---------- 反思：累计重要度过了阈值就该回顾一下、提炼出更高层的认识（同样出自那篇论文） ----------
def test_importance_accumulates_and_crosses_reflection_threshold():
    s = MemoryStore()
    assert not s.should_reflect()
    for _ in range(15):
        s.add("琐事", 10, NOW)  # 15 * 10 = 150，正好到阈值
    assert s.should_reflect()


def test_mark_reflected_resets_the_counter():
    s = MemoryStore()
    for _ in range(20):
        s.add("琐事", 10, NOW)
    assert s.should_reflect()
    s.mark_reflected()
    assert not s.should_reflect()
    assert s.importance_since_reflection == 0.0


def test_custom_threshold_overrides_default():
    s = MemoryStore()
    s.add("小事", 5, NOW)
    assert not s.should_reflect(threshold=10.0)
    assert s.should_reflect(threshold=5.0)


def test_save_and_load_roundtrip_preserves_reflection_counter(tmp_path):
    s = MemoryStore()
    s.add("琐事", 7, NOW)
    path = tmp_path / "with_counter.json"
    s.save(path)
    loaded = MemoryStore.load(path)
    assert loaded.importance_since_reflection == 7.0


def test_load_old_file_without_reflection_field_defaults_to_zero(tmp_path):
    """兼容加反思之前存的记忆文件：没有这个字段，读出来该是 0，不该报错、也不该凭空触发反思。"""
    import json

    path = tmp_path / "old_format.json"
    path.write_text(
        json.dumps({"met": [], "memories": [{"text": "老格式", "time": NOW, "importance": 9, "people": []}]}),
        encoding="utf-8",
    )
    loaded = MemoryStore.load(path)
    assert loaded.importance_since_reflection == 0.0
    assert not loaded.should_reflect()


# ---------- 主动分享：挑"值得跟这个人说、而且还没说过"的事 ----------
def share_store() -> MemoryStore:
    s = MemoryStore()
    s.add("集市的米价涨了三成", 8, NOW)  # 够重要、跟 bob 无关
    s.add("今天天气不错", 2, NOW)  # 不够重要
    s.add("Bob 说他很忙", 9, NOW, {"bob"})  # bob 自己就在这条记忆里
    return s


def test_shareable_filters_by_importance_and_involvement():
    picked = share_store().shareable("bob", NOW, k=5)
    assert [m.text for m in picked] == ["集市的米价涨了三成"]


def test_shareable_threshold_is_inclusive():
    s = MemoryStore()
    s.add("刚好到阈值", SHARE_MIN_IMPORTANCE, NOW)
    s.add("差一点", SHARE_MIN_IMPORTANCE - 1, NOW)
    assert [m.text for m in s.shareable("bob", NOW, k=5)] == ["刚好到阈值"]


def test_marked_as_told_is_not_offered_again():
    s = share_store()
    picked = s.shareable("bob", NOW, k=1)
    s.mark_told(picked[0], "bob")
    assert s.shareable("bob", NOW, k=1) == []


def test_told_to_is_tracked_per_person():
    """跟 bob 讲过不等于跟 carol 也讲过；而且"Bob 说他很忙"这条对 carol 是可以讲的
    （对 bob 自己不行）——这正是八卦该有的样子。"""
    s = share_store()
    s.mark_told(s.shareable("bob", NOW, k=1)[0], "bob")
    carol_can_hear = [m.text for m in s.shareable("carol", NOW, k=5)]
    assert "集市的米价涨了三成" in carol_can_hear
    assert "Bob 说他很忙" in carol_can_hear


def test_shareable_ranked_by_score_and_respects_k():
    s = MemoryStore()
    s.add("旧的重要事", 7, NOW - 5000)
    s.add("新的重要事", 7, NOW)
    assert s.shareable("bob", NOW, k=1)[0].text == "新的重要事"
    assert len(s.shareable("bob", NOW, k=2)) == 2
    assert s.shareable("bob", NOW, k=0) == []


# ---------- 传播代数：消息传得越远越不当真 ----------
def test_retold_importance_discounts_once_per_telling():
    assert retold_importance(9) == round(9 * HOP_IMPORTANCE_DECAY)


def test_retold_importance_never_reaches_zero():
    """再怎么传得远，它毕竟还是件"我知道的事"，不该被压成 0 直接消失。"""
    assert retold_importance(1) == 1


def test_retold_importance_clamps_out_of_range_input():
    assert retold_importance(99) == retold_importance(10)
    assert retold_importance(-5) == retold_importance(1)


def test_hop_is_metadata_and_does_not_discount_by_itself():
    """打折由写入方按 retold_importance() 一次一次折下来（见 agent._remember），
    add() 要是再按代数折一遍，就成了双重折扣。"""
    s = MemoryStore()
    s.add("传了很多手", 9, NOW, hop=5)
    assert s.memories[0].importance == 9
    assert s.memories[0].hop == 5


def test_hop_is_clamped_to_non_negative():
    s = MemoryStore()
    s.add("负数代数", 9, NOW, hop=-3)
    assert s.memories[0].hop == 0
    assert s.memories[0].importance == 9


def test_rumor_importance_decays_until_nobody_bothers_passing_it_on():
    """有意思的涌现性质：消息会自己"传死"——转述几次之后重要度掉到分享门槛以下，
    就没人再往下传了。这不是硬写的规则，是衰减系数和分享门槛两个数凑在一起的结果。

    注意这条递推是"每转述一次折一次"，不是按代数一次性指数打折：听者记的是"转述者
    当时觉得这事有多重要"再打一折，而转述者那个数本身也是这么一路折下来的。"""
    seq, importance = [9], 9
    while importance >= SHARE_MIN_IMPORTANCE:
        importance = retold_importance(importance)
        seq.append(importance)
    assert seq == [9, 7, 6, 5]
    assert seq[-1] < SHARE_MIN_IMPORTANCE


# ---------- 落盘 ----------
def test_hop_and_told_to_survive_roundtrip(tmp_path):
    s = MemoryStore()
    s.add("集市的米价涨了三成", 9, NOW, hop=1)
    s.mark_told(s.memories[0], "bob")
    path = tmp_path / "m.json"
    s.save(path)
    loaded = MemoryStore.load(path)
    assert loaded.memories[0].hop == 1
    assert loaded.memories[0].told_to == frozenset({"bob"})
    assert loaded.shareable("bob", NOW, k=1) == []  # 读回来之后照样不会再讲给 bob


def test_old_memory_file_without_new_fields_still_loads(tmp_path):
    """老的记忆文件没有 hop / told_to 字段，要能读进来并退化成"第一手、没跟谁讲过"，
    跟加这两个功能之前的行为一致。"""
    s = MemoryStore()
    s.add("老记忆", 7, NOW)
    path = tmp_path / "m.json"
    s.save(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    for d in raw["memories"]:
        d.pop("hop", None)
        d.pop("told_to", None)
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    loaded = MemoryStore.load(path)
    assert len(loaded.memories) == 1
    assert loaded.memories[0].hop == 0
    assert loaded.memories[0].told_to == frozenset()
