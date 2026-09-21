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
