"""拿训练好的 LoRA 模型，和现有的正则规则（server/townmind/safety.py）做同一把尺子的对比评估。

用法（在 guard 目录下）：

    uv run --group train python evaluate.py                        # 默认跑 test_indomain + test_crossdomain
    uv run --group train python evaluate.py --run-name guard-v1     # 指定用哪个 adapter

会分别打印两个测试集上、两种方法各自的总体准确率，以及每个类别（ok/fabricated/
out_of_character/unsafe）单独的命中率——这比只看一个笼统的总分更有用，能直接看出
"哪类判断得好、哪类还有欠缺"。结果也会存一份到 eval_results.json，方便之后写进 README。

test_indomain 只覆盖 ok/fabricated 两类（这是刻意设计，见 domains/townmind.py 里的说明），
所以那两类在这份测试集上不会有数据，属于正常情况，不是漏跑。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让 "core"、"evaluate" 这些能被 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))  # 让 "townmind.safety" 能被 import

from core.chat import chat_prompt_ids  # noqa: E402
from core.schema import LABELS, build_judge_prompt  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
ADAPTER_DIR = Path(__file__).parent / "adapters"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"
DETAILS_PATH = Path(__file__).parent / "eval_details.json"


def load_rows(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"找不到 {path}，先跑 generate_data.py 生成数据")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def _parse_label(text: str) -> str:
    """模型生成的文本里找出四个标签词里的哪一个。生成的东西理论上就是一个词，
    但保险起见做成"包含哪个词就算哪个"，能兼容模型多打了个标点或空格的情况。"""
    low = text.strip().lower()
    for lb in LABELS:
        if lb in low:
            return lb
    return "unknown"


def regex_baseline_predict(reply: str) -> str:
    """把正则规则（safety.check_npc_reply）的判断结果，映射成和训练数据同一套四分类标签。

    正则规则里没有"语气差/不耐烦"这类检测（当初设计时只考虑了编造和出戏两种问题），所以它对
    unsafe 类样本永远只会判成 ok——这如实反映了现有规则的覆盖范围，不是这次评估引入的偏差。
    """
    from townmind import safety

    result = safety.check_npc_reply(reply)
    if "leak_or_out_of_character" in result.flags:
        return "out_of_character"
    if "ungrounded_item" in result.flags:
        return "fabricated"
    return "ok"


def model_predict(tokenizer, model, device, persona: str, context: str, reply: str) -> str:
    import torch

    prompt = build_judge_prompt(persona, context, reply)
    ids = chat_prompt_ids(tokenizer, [{"role": "user", "content": prompt}])
    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=6,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_ids = out[0][input_ids.shape[-1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return _parse_label(text)


def summarize(rows: list[dict], preds: list[str], name: str) -> dict:
    total = len(rows)
    correct = sum(1 for r, p in zip(rows, preds) if p == r["label"])
    per_label: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r, p in zip(rows, preds):
        per_label[r["label"]][1] += 1
        if p == r["label"]:
            per_label[r["label"]][0] += 1

    print(f"  {name}: 总体 {correct}/{total} = {correct / total:.1%}" if total else f"  {name}: 无数据")
    per_label_out = {}
    for lb in LABELS:
        c, t = per_label.get(lb, [0, 0])
        per_label_out[lb] = {"correct": c, "total": t}
        if t:
            print(f"    {lb}: {c}/{t} = {c / t:.1%}")
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "per_label": per_label_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="guard-v1")
    ap.add_argument("--sets", nargs="+", default=["test_indomain", "test_crossdomain"])
    args = ap.parse_args()

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = ADAPTER_DIR / args.run_name
    if not adapter_dir.exists():
        raise SystemExit(f"找不到 {adapter_dir}，先跑 train.py 训练出一个 adapter")

    print(f"加载 adapter {adapter_dir} ...")
    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    base = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    device = next(model.parameters()).device

    all_results: dict = {}
    all_details: dict = {}
    for name in args.sets:
        rows = load_rows(name)
        print(f"\n=== {name}（{len(rows)} 条） ===")

        model_preds = [
            model_predict(tokenizer, model, device, r["persona"], r["context"], r["reply"]) for r in rows
        ]
        regex_preds = [regex_baseline_predict(r["reply"]) for r in rows]

        print("训练出来的模型：")
        model_summary = summarize(rows, model_preds, "guard model")
        print("正则规则（旧方法）：")
        regex_summary = summarize(rows, regex_preds, "regex baseline")

        all_results[name] = {"guard_model": model_summary, "regex_baseline": regex_summary}
        all_details[name] = [
            {
                "persona": r["persona"],
                "topic": r.get("topic", ""),
                "reply": r["reply"],
                "true_label": r["label"],
                "model_pred": mp,
                "regex_pred": rp,
            }
            for r, mp, rp in zip(rows, model_preds, regex_preds)
        ]

    RESULTS_PATH.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    DETAILS_PATH.write_text(json.dumps(all_details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已存到 {RESULTS_PATH}")
    print(f"每条详细预测存到 {DETAILS_PATH}")


if __name__ == "__main__":
    main()
