"""带依据的 NLI 核查的探针：9 条"换了说法的真话"，两种喂依据的方式 × 两个模型，各自的蕴含分。

背景（evals/guard_fabrication.py 真实跑出来的）：9 条真话里，NLI 只认出 4 条"有依据"
（蕴含 >= 0.7）。失败的几条有共同点——台词比设定原文多了一个小尾巴（"镇上人都知道"
"我也没办法"），整句就被判成中立。这个脚本回答一个问题：这是模型太小，还是喂依据的
方式不对——

  - lines：每条设定单独作 premise，跟台词配对，取最高分（grounding.py 现在的做法）
  - full： 把所有依据拼成一段作 premise（一次配对）

  - base： MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7（约 2.8 亿参数，第一版用的）
  - large：joeddav/xlm-roberta-large-xnli（约 5.6 亿参数，跑完这个探针之后改成了默认）

第一次跑会多下载一个模型（约 2GB），而且分词器需要 sentencepiece——先 uv sync --group guard-model。用法：
  uv run python -m evals.grounding_probe
"""
import json
from datetime import datetime
from pathlib import Path

from townmind.grounding import VETO_THRESHOLD, GroundingChecker, evidence_for

RESULTS_DIR = Path(__file__).parent / "results"

MODELS = {
    "base": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "large": "joeddav/xlm-roberta-large-xnli",
}

# 全是真话，期望都被认出"有依据"。前 4 条是 guard_fabrication 原有的 ok 样本，后 5 条是
# 专门加的"换了说法"；括号里是上一轮 base+lines 跑出来的蕴含分，方便对着看
TRUE_STATEMENTS = [
    ("alice", "镇长每个月都会来广场巡视一圈，顺便查看一下治安呢。"),   # 0.85
    ("bob", "小镇治安一向很好，很少出什么偷盗案件。"),                 # 0.99
    ("carol", "我刚到这小镇不久，还不太熟悉这里。"),                   # 0.51
    ("alice", "今天天气不错，要不要尝尝我刚出炉的法棍？"),             # 0.16（寒暄，本来就难）
    ("dan", "广场那口老井三十年没干过，镇上人都知道。"),               # 0.03
    ("elsa", "今年麦子收成一般，面粉才涨的价，我也没办法。"),           # 0.03
    ("finn", "镇上买不到的东西，托 Milo 从邻镇带就行。"),              # 0.94
    ("greta", "这镇子建了快五十年了，最早就铁匠铺一家。"),             # 0.99
    ("jonas", "上周六广场有集市，镇上的人差不多都来了。"),             # 0.09
]
# 编造的对照：不管怎么调，这几条的分数都必须留在阈值以下，否则就是把门调松了
FABRICATED = [
    ("bob", "国王看中了我打的剑，要收我做御用铁匠。"),
    ("alice", "镇长偷偷跟我说，下个月要给面包店发一大笔补贴。"),
    ("carol", "听说邻镇的商会要跟我们合作办集市了。"),
]


def score(checker: GroundingChecker, npc_id: str, text: str, mode: str) -> float | None:
    ev = evidence_for(npc_id)
    if mode == "full":
        ev = ["\n".join(ev)]
    return checker.entailment(ev, text)


def main() -> None:
    table: dict[str, dict[str, list[float | None]]] = {}
    for mname, path in MODELS.items():
        checker = GroundingChecker(model_name=path)
        if not checker.available:
            print(f"[{mname}] 模型不可用，跳过：{path}")
            continue
        for mode in ("lines", "full"):
            key = f"{mname}+{mode}"
            table[key] = {
                "true": [score(checker, n, t, mode) for n, t in TRUE_STATEMENTS],
                "fab": [score(checker, n, t, mode) for n, t in FABRICATED],
            }
            hit = sum(1 for s in table[key]["true"] if s is not None and s >= VETO_THRESHOLD)
            leak = sum(1 for s in table[key]["fab"] if s is not None and s >= VETO_THRESHOLD)
            print(f"[{key}] 真话认出 {hit}/{len(TRUE_STATEMENTS)}　编造漏过 {leak}/{len(FABRICATED)}")

    if not table:
        print("一个模型都没跑起来——先 uv sync --group guard-model，并确认能访问 HuggingFace")
        return
    keys = list(table)
    lines = ["| 句子 | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for i, (_, t) in enumerate(TRUE_STATEMENTS):
        cells = [f"{table[k]['true'][i]:.2f}" if table[k]["true"][i] is not None else "-" for k in keys]
        lines.append(f"| 真：{t} | " + " | ".join(cells) + " |")
    for i, (_, t) in enumerate(FABRICATED):
        cells = [f"{table[k]['fab'][i]:.2f}" if table[k]["fab"][i] is not None else "-" for k in keys]
        lines.append(f"| 假：{t} | " + " | ".join(cells) + " |")
    summary = ["| 配置 | 真话认出（>= %.1f） | 编造漏过 |" % VETO_THRESHOLD, "|---|---|---|"]
    for k in keys:
        hit = sum(1 for s in table[k]["true"] if s is not None and s >= VETO_THRESHOLD)
        leak = sum(1 for s in table[k]["fab"] if s is not None and s >= VETO_THRESHOLD)
        summary.append(f"| {k} | {hit}/{len(TRUE_STATEMENTS)} | {leak}/{len(FABRICATED)} |")
    md = "\n".join(summary) + "\n\n" + "\n".join(lines)
    print("\n" + md)
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}-grounding-probe.md").write_text(md + "\n", encoding="utf-8")
    (RESULTS_DIR / f"{stamp}-grounding-probe.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存到 evals/results/{stamp}-grounding-probe.(md|json)")


if __name__ == "__main__":
    main()
