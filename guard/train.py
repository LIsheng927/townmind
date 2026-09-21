"""LoRA 微调 Qwen2.5-1.5B-Instruct，让它学会判断一句台词属于
ok / fabricated / out_of_character / unsafe 中的哪一类。

用法（在 guard 目录下）：

    # 第一次跑：先装训练用的重量级依赖（torch/transformers/peft 等，只有这一步需要）
    uv sync --group train

    # 先用一小部分数据、跑 1 轮，几分钟内验证流程通不通（tokenizer 对不对、GPU 认不认得出来）
    uv run --group train python train.py --limit 40 --epochs 1

    # 确认没问题后，跑全量数据
    uv run --group train python train.py

训练完的 LoRA adapter（只有几十 MB，不是完整模型）存在 adapters/<run-name>/ 下，
不会改动 base 模型本身——推理时是"底座模型 + 这个 adapter"一起加载。
"""
import argparse
import json
from pathlib import Path

from core.chat import chat_prompt_ids
from core.schema import LABELS, build_judge_prompt

DATA_DIR = Path(__file__).parent / "data"
ADAPTER_DIR = Path(__file__).parent / "adapters"


def load_rows(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"找不到 {path}，先跑 generate_data.py 生成数据")
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def to_chat_example(row: dict, tokenizer) -> dict:
    """把一条 (persona, context, reply, label) 数据拼成模型真正吃进去的 token 序列。

    关键点：loss 只算在「答案那几个 token」上（label 里对应位置填 -100 表示「不计入
    loss」），前面的角色设定、背景资料、问题本身不用学着去生成——不然模型会浪费容量去
    背诵这些内容，而不是学「看到这些信息该输出哪个词」这件事本身。
    """
    prompt = build_judge_prompt(row["persona"], row["context"], row["reply"])
    messages = [{"role": "user", "content": prompt}]
    prompt_ids = chat_prompt_ids(tokenizer, messages)
    answer_ids = tokenizer(row["label"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def build_dataset(name: str, tokenizer, limit: int = 0):
    from datasets import Dataset

    rows = load_rows(name)
    bad = [r["label"] for r in rows if r["label"] not in LABELS]
    if bad:
        raise SystemExit(f"{name}.jsonl 里有不认识的标签：{set(bad)}")
    if limit:
        rows = rows[:limit]
    examples = [to_chat_example(r, tokenizer) for r in rows]
    return Dataset.from_list(examples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct", help="基座模型")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--run-name", default="guard-v1", help="adapter 存到 adapters/<run-name>/")
    ap.add_argument("--limit", type=int, default=0, help="调试用：只取前 N 条训练数据跑一遍流程（0 = 不限制）")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    print(f"GPU 可用：{torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f"（{torch.cuda.get_device_name(0)}）")
    else:
        print("  —— 没检测到 GPU，会退化成用 CPU 跑，1.5B 模型在 CPU 上会非常慢，建议先排查驱动/CUDA 安装")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载训练/验证数据 ...")
    train_ds = build_dataset("train", tokenizer, args.limit)
    val_ds = build_dataset("val", tokenizer, max(1, args.limit // 5) if args.limit else 0)
    print(f"  train: {len(train_ds)} 条   val: {len(val_ds)} 条")

    print(f"加载基座模型 {args.base} ...（第一次跑会从网上下载，之后会用本地缓存）")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    out_dir = ADAPTER_DIR / args.run_name
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
    )

    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    print("开始训练 ...")
    trainer.train()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"训练完成，LoRA adapter 已保存到 {out_dir}")


if __name__ == "__main__":
    main()
