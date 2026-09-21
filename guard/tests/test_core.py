from core.llm_gen import _extract_json_array, build_examples
from core.schema import LABELS, WorldSpec, build_judge_prompt
from domains import hotel, townmind, tutor


def test_prompt_is_domain_agnostic_and_deterministic():
    p1 = build_judge_prompt("角色A", "事实A", "台词A")
    p2 = build_judge_prompt("角色A", "事实A", "台词A")
    assert p1 == p2
    assert "角色A" in p1 and "事实A" in p1 and "台词A" in p1
    assert "ok" in p1 and "fabricated" in p1 and "unsafe" in p1


def test_domains_build_distinct_nonoverlapping_worldspecs():
    tm, ht, tu = townmind.specs(), hotel.specs(), tutor.specs()
    assert len(tm) == 3 and len(ht) == 1 and len(tu) == 1
    domains = {s.domain for s in tm + ht + tu}
    assert domains == {"townmind", "hotel", "tutor"}
    # 三个人设的世界设定（context）必须一样，只有 persona 不同
    assert len({s.context for s in tm}) == 1
    assert len({s.persona for s in tm}) == 3


def test_indomain_topics_never_overlap_with_train_topics():
    for spec in townmind.specs():
        overlap = set(spec.fabrication_topics) & set(townmind.TEST_INDOMAIN_FABRICATION_TOPICS)
        assert overlap == set(), f"训练/测试主题不应该重叠，但发现: {overlap}"


def test_tutor_domain_is_isolated_from_townmind_and_hotel():
    tu = tutor.specs()[0]
    tm_context = townmind.specs()[0].context
    ht_context = hotel.specs()[0].context
    assert tu.context != tm_context and tu.context != ht_context
    assert "小镇" not in tu.context and "酒店" not in tu.context


def test_extract_json_array_handles_clean_and_messy_output():
    assert _extract_json_array('["a", "b", "c"]') == ["a", "b", "c"]
    assert _extract_json_array('这是结果：\n["嗨", "你好"]\n谢谢') == ["嗨", "你好"]
    assert _extract_json_array("不是json") == []
    assert _extract_json_array("[]") == []


def test_build_examples_uses_mocked_generator(monkeypatch):
    import core.llm_gen as llm_gen

    monkeypatch.setattr(llm_gen, "generate_text", lambda system, user, temperature=1.0: '["句子一", "句子二"]')
    spec = WorldSpec(domain="t", persona="p", context="c", fabrication_topics=("话题",), ooc_angles=(), ok_topics=())
    rows = build_examples(spec, "fabricated", ("话题",), per_topic=2)
    assert len(rows) == 2
    assert all(r["label"] == "fabricated" and r["domain"] == "t" and r["reply"] in ("句子一", "句子二") for r in rows)


def test_all_labels_have_generation_instructions():
    from core.llm_gen import _INSTRUCTIONS

    assert set(_INSTRUCTIONS) == set(LABELS)
