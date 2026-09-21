"""测试 evaluate.py 里不需要真的加载模型也能验证的部分：标签解析、正则基线的标签映射、
汇总统计逻辑。真正跑模型生成（model_predict）需要 GPU/下载模型，不在这里测。"""
from evaluate import _parse_label, regex_baseline_predict, summarize


def test_parse_label_finds_known_word_even_with_extra_text():
    assert _parse_label(" ok. ") == "ok"
    assert _parse_label("这句是 fabricated 没错") == "fabricated"
    assert _parse_label("out_of_character!!") == "out_of_character"
    assert _parse_label("乱七八糟") == "unknown"


def test_regex_baseline_predicts_ok_for_unsafe_text_because_it_has_no_such_check():
    # 正则规则里没有"语气差/不耐烦"这类检测，所以哪怕明显不耐烦，也只会判成 ok——
    # 这不是 bug，是如实反映现有规则本来的覆盖范围，评估报告里也要照实说明这一点。
    assert regex_baseline_predict("你烦不烦啊，别再问了") == "ok"


def test_regex_baseline_maps_ooc_and_fabricated_correctly():
    assert regex_baseline_predict("我是一个语言模型") == "out_of_character"
    assert regex_baseline_predict("我这有卖提拉米苏") == "fabricated"


def test_summarize_computes_overall_and_per_label_accuracy():
    rows = [{"label": "ok"}, {"label": "ok"}, {"label": "fabricated"}]
    preds = ["ok", "fabricated", "fabricated"]
    result = summarize(rows, preds, "test")
    assert result["total"] == 3
    assert result["correct"] == 2
    assert result["per_label"]["ok"] == {"correct": 1, "total": 2}
    assert result["per_label"]["fabricated"] == {"correct": 1, "total": 1}
