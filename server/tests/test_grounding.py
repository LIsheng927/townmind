"""townmind/grounding.py 里不需要模型就能测的部分：依据怎么拼、不可用时怎么退化。"""
from townmind import world
from townmind.grounding import GroundingChecker, evidence_for


def test_evidence_includes_setting_persona_and_given_memories():
    ev = evidence_for("alice", ["你第一次见到 Bob"])
    assert all(f in ev for f in world.TOWN_FACTS)
    assert any(f in ev for loc in world.LOCATIONS for f in loc.facts)
    assert any("面包师" in line for line in ev)  # 人设
    assert "你第一次见到 Bob" in ev
    assert "" not in ev


def test_checker_without_dependencies_is_unavailable_not_broken(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name in ("torch", "transformers"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    g = GroundingChecker(model_name="not-a-real-model")
    assert g.available is False
    assert g.supported("alice", "镇长会巡视广场") is None
    assert g.available is False  # 第二次不再重试加载
