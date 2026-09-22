from townmind.memory import MemoryStore, format_age

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
