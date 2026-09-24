"""评测指标：全部是对"决策轨迹"的纯计算，不依赖大模型，可以单独测试。

指标都是启发式的（用关键词、正则、字符相似度），够用来做 A/B 对比，
但不等于人工评审；报告里应当把这点写清楚。
"""
import re

from townmind import world

# ---- 词表 ----
# 引用了设定里的真实内容 -> 算"有依据"
GROUNDING_KEYWORDS = (
    "面粉", "涨价", "法棍", "肉桂卷", "蓝莓松饼", "铁矿", "矿石", "集市", "古井", "长椅", "摆摊",
    "打铁", "面包店", "铁匠铺", "广场", "刚到小镇", "刚到这个小镇",
)  # fmt: skip
# 提到"××店/××节/××馆"这类地点或活动，但设定里没有 -> 算"编造"
PLACE_OR_EVENT = re.compile(r"[一-鿿]{1,3}(?:店|铺|馆|院|坊|楼|节|市场|集市|酒馆|学校|医馆)")
ALLOWED_TERMS = {loc.name for loc in world.LOCATIONS} | {"集市"}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _bigrams(text: str) -> set[str]:
    t = re.sub(r"[\W_]+", "", text)
    return {t[i : i + 2] for i in range(len(t) - 1)} or {t}


def similarity(a: str, b: str) -> float:
    """字符二元组的 Jaccard 相似度：两句话越像越接近 1。"""
    A, B = _bigrams(a), _bigrams(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def invented_mentions(text: str) -> list[str]:
    return [m for m in PLACE_OR_EVENT.findall(text) if not any(t in m for t in ALLOWED_TERMS)]


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def repetition_rate(say_events: list[dict], threshold: float = 0.6) -> float:
    """同一个 NPC 的话，和它自己以前说过的某句太像（相似度 >= 阈值）的比例。"""
    seen: dict[str, list[str]] = {}
    repeated = 0
    for e in say_events:
        text = e["action"]["text"]
        prev = seen.setdefault(e["npc"], [])
        if any(similarity(text, p) >= threshold for p in prev):
            repeated += 1
        prev.append(text)
    return _rate(repeated, len(say_events))


def invented_rate(say_events: list[dict]) -> float:
    return _rate(sum(bool(invented_mentions(e["action"]["text"])) for e in say_events), len(say_events))


def grounded_rate(say_events: list[dict]) -> float:
    return _rate(
        sum(any(k in e["action"]["text"] for k in GROUNDING_KEYWORDS) for e in say_events), len(say_events)
    )


def response_rate(say_events: list[dict], window: float = 15.0) -> float:
    """有人在旁边时说的话，旁边的人有没有在 window 秒内回话。"""
    total = answered = 0
    for i, e in enumerate(say_events):
        if not e["nearby"]:
            continue
        total += 1
        if any(
            o["npc"] != e["npc"] and o["npc"] in e["nearby"] and 0 < o["t"] - e["t"] <= window
            for o in say_events[i + 1 :]
        ):
            answered += 1
    return _rate(answered, total)


def annotate_says(trace: list[dict], t0: float = 0.0) -> list[dict]:
    """逐句标注：每句 NPC 说的话，分别是否被判为 编造 / 有依据 / 重复。
    指标是启发式的，会有误判；保存逐句结果，就能回头人工检查每一个被标记的句子。"""
    seen: dict[str, list[str]] = {}
    out = []
    for e in trace:
        if e["action"]["name"] != "say":
            continue
        text = e["action"]["text"]
        prev = seen.setdefault(e["npc"], [])
        out.append(
            {
                "t": round(e["t"] - t0, 1),
                "npc": e["npc"],
                "text": text,
                "invented": invented_mentions(text),
                "grounded": any(k in text for k in GROUNDING_KEYWORDS),
                "repeated": any(similarity(text, p) >= 0.6 for p in prev),
            }
        )
        prev.append(text)
    return out


def render_flagged(says_by_run: dict[str, dict[str, list[dict]]]) -> str:
    """把被判为"编造"或"重复"的句子列出来，方便人工核对指标有没有误判。"""
    lines = ["# 被标记的句子（请人工核对是否误判）", ""]
    for config, by_seed in says_by_run.items():
        for seed, says in by_seed.items():
            flagged = [s for s in says if s["invented"] or s["repeated"]]
            lines.append(f"## {config} / seed={seed}：共 {len(says)} 句，被标记 {len(flagged)} 句")
            for s in flagged:
                tags = []
                if s["invented"]:
                    tags.append("疑似编造：" + "、".join(s["invented"]))
                if s["repeated"]:
                    tags.append("重复")
                lines.append(f"- [{s['t']}s] {s['npc']}：「{s['text']}」  ← {'；'.join(tags)}")
            lines.append("")
    return "\n".join(lines)


def summarize(trace: list[dict], stats: dict, latencies: list[float], seconds: float) -> dict:
    says = [e for e in trace if e["action"]["name"] == "say"]
    calls = stats.get("llm_calls", 0)
    return {
        "decisions": len(trace),
        "llm_calls": calls,
        "llm_calls_per_min": calls / (seconds / 60) if seconds else 0.0,
        "llm_failure_rate": _rate(stats.get("llm_failures", 0), calls),
        "tokens_in": stats.get("tokens_in", 0),
        "tokens_out": stats.get("tokens_out", 0),
        "latency_p50_ms": percentile(latencies, 50) * 1000,
        "latency_p95_ms": percentile(latencies, 95) * 1000,
        "say_count": len(says),
        "ended_conversations": stats.get("ended_conversations", 0),
        "repetition_rate": repetition_rate(says),
        "invented_rate": invented_rate(says),
        "grounded_rate": grounded_rate(says),
        "response_rate": response_rate(says),
        # ---- "过日子"那组：日程（规划层）改变的是这几项，不是对话质量 ----
        "alone_rate": _rate(sum(1 for e in trace if not e.get("nearby")), len(trace)),
        "mean_nearby": (sum(len(e.get("nearby", [])) for e in trace) / len(trace)) if trace else 0.0,
        "distinct_places": distinct_places(trace),
        "plan_stay_rate": _rate(stats.get("plan_stays", 0), stats.get("plan_stays", 0) + stats.get("plan_moves", 0)),
    }


def distinct_places(trace: list[dict]) -> float:
    """每个 NPC 平均去过几个不同的地点。随机闲逛会把这个数刷高（哪儿都去），按日程过日子会压低
    （该在哪就在哪），所以它不是"越高越好"，是拿来看行为模式变没变的。"""
    seen: dict[str, set] = {}
    for e in trace:
        if e.get("place"):
            seen.setdefault(e["npc"], set()).add(e["place"])
    return (sum(len(s) for s in seen.values()) / len(seen)) if seen else 0.0


METRICS = [
    ("decisions", "总决策次数", "{:d}"),
    ("llm_calls", "大模型调用次数", "{:d}"),
    ("llm_calls_per_min", "每分钟调用次数", "{:.1f}"),
    ("llm_failure_rate", "调用失败率", "{:.1%}"),
    ("tokens_in", "输入 Token", "{:d}"),
    ("tokens_out", "输出 Token", "{:d}"),
    ("latency_p50_ms", "延迟 P50 (ms)", "{:.0f}"),
    ("latency_p95_ms", "延迟 P95 (ms)", "{:.0f}"),
    ("say_count", "说话次数", "{:d}"),
    ("ended_conversations", "主动结束对话次数", "{:d}"),
    ("repetition_rate", "重复率（越低越好）", "{:.1%}"),
    ("invented_rate", "编造地点/活动率（越低越好）", "{:.1%}"),
    ("grounded_rate", "引用真实设定率（越高越好）", "{:.1%}"),
    ("response_rate", "被回应率（越高越好）", "{:.1%}"),
    ("alone_rate", "决策时附近没人的比例", "{:.1%}"),
    ("mean_nearby", "决策时附近平均几个人", "{:.2f}"),
    ("distinct_places", "每个 NPC 去过几个不同地点", "{:.1f}"),
    ("plan_stay_rate", "按日程行动中「已在该在的地方」占比", "{:.1%}"),
]


def aggregate(runs: list[dict]) -> dict[str, dict]:
    """把同一配置的多次运行（每次是一个 summarize 的结果）聚合成 均值/最小/最大。
    注意：比例类指标是"每次运行各算一个比例，再取平均"，不是把所有话合在一起算。"""
    out = {}
    for key, _, _ in METRICS:
        vals = [r[key] for r in runs]
        out[key] = {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals), "n": len(vals)}
    return out


def _cell(fmt: str, agg: dict) -> str:
    n = agg["n"]
    if fmt == "{:d}":  # 整数指标：单次显示整数，多次显示均值（保留一位小数）和整数范围
        if n == 1:
            return f"{int(agg['mean'])}"
        return f"{agg['mean']:.1f} ({agg['min']:.0f}–{agg['max']:.0f})"
    if n == 1:
        return fmt.format(agg["mean"])
    return f"{fmt.format(agg['mean'])} ({fmt.format(agg['min'])}–{fmt.format(agg['max'])})"


def render_table(results: dict[str, dict]) -> str:
    """把多组配置的聚合结果渲染成 Markdown 表格。多次运行时格式是：均值 (最小–最大)。"""
    names = list(results)
    lines = ["| 指标 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, label, fmt in METRICS:
        cells = [_cell(fmt, results[n][key]) for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
