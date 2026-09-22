"""GuardModel 是彻底可选的一层：没装 torch/transformers/peft、或者没有训练好的 adapter，
都应该优雅地"不可用"，而不是抛异常——这里不装真的模型（几个 GB，CI 也没有 GPU），只测
"缺东西时该怎么表现"和"prompt 拼得对不对"这两件事；真正的推理准确率由 guard/evaluate.py
在有真实权重、真实依赖的环境里单独验证（见 guard/README.md 的实测结果）。"""
from pathlib import Path

from townmind import guard_model as gm
from townmind import world


def test_full_world_context_matches_world_data():
    ctx = gm._full_world_context()
    for loc in world.LOCATIONS:  # 每个地点都要在里面，不是只有最早那三个
        assert loc.name in ctx, loc.name
    assert "面粉涨价了" in ctx  # 地点自己的 facts 也要在里面，不能只拼地点描述
    assert world.TOWN_FACTS[0] in ctx  # TOWN_FACTS 也要在里面（从设定里取，改文案不用回来改测试）


def test_persona_for_known_npc_matches_training_time_format():
    p = gm._persona_for("alice")
    assert p.startswith("你是游戏小镇里的 NPC「Alice」。")
    assert "面包店" in p  # 工作地点要带上，跟 guard/domains/townmind.py 拼法一致


def test_persona_for_unknown_npc_falls_back_to_default():
    p = gm._persona_for("nobody")
    assert "路人" in p


def test_unavailable_when_adapter_dir_does_not_exist():
    g = gm.GuardModel(adapter_dir=Path("/definitely/does/not/exist"))
    assert g.available is False
    assert g.classify("alice", "随便一句话") is None


def test_unavailable_is_sticky_and_does_not_raise_on_repeated_calls():
    g = gm.GuardModel(adapter_dir=Path("/definitely/does/not/exist"))
    assert g.classify("alice", "第一句") is None
    assert g.classify("alice", "第二句") is None
    assert g._unavailable is True


def test_loads_real_guard_core_schema_and_builds_matching_prompt():
    """guard/core/schema.py 是训练、评测（guard/evaluate.py）、这里三处共用的"标签 + prompt
    模板"唯一来源。这个测试不需要 torch，只验证"按路径加载"这条路径本身没坏、加载出来的
    标签集和 prompt 格式符合预期——真正防止三处 prompt 格式跑偏的，是它们永远读同一份文件，
    而不是各自维护一份可能会漂移的拷贝。"""
    schema = gm._load_module_from_file("_test_guard_schema", gm._GUARD_CORE_DIR / "schema.py")
    assert schema.LABELS == ("ok", "fabricated", "out_of_character", "unsafe")
    prompt = schema.build_judge_prompt("我是A", "背景资料", "一句话")
    assert "我是A" in prompt and "背景资料" in prompt and "一句话" in prompt
    assert "ok / fabricated / out_of_character / unsafe" in prompt


def test_loads_real_guard_core_chat_helper():
    chat = gm._load_module_from_file("_test_guard_chat", gm._GUARD_CORE_DIR / "chat.py")
    assert callable(chat.chat_prompt_ids)
