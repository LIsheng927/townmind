from generate_data import _dedup, _gen_for_spec
from core.schema import WorldSpec


def test_dedup_drops_repeats_and_overlong_and_empty():
    rows = [
        {"domain": "d", "reply": "你好", "label": "ok"},
        {"domain": "d", "reply": "你好", "label": "ok"},  # 重复
        {"domain": "d", "reply": "", "label": "ok"},  # 空
        {"domain": "d", "reply": "长" * 200, "label": "ok"},  # 太长，八成是模型没听指令
        {"domain": "d", "reply": "再见", "label": "ok"},
    ]
    out = _dedup(rows)
    assert [r["reply"] for r in out] == ["你好", "再见"]


def test_gen_for_spec_covers_all_four_labels(monkeypatch):
    import core.llm_gen as llm_gen

    monkeypatch.setattr(llm_gen, "generate_text", lambda system, user, temperature=1.0: '["示例句子"]')
    spec = WorldSpec(domain="t", persona="p", context="c", fabrication_topics=("a",), ooc_angles=("b",), ok_topics=("c",))
    rows = _gen_for_spec(spec, per_topic=1)
    assert {r["label"] for r in rows} == {"ok", "fabricated", "out_of_character", "unsafe"}
