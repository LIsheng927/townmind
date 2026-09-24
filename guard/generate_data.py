"""生成训练/测试数据。用法（在 guard 目录下）：

  uv run python generate_data.py                      # 用默认数量生成全部四份数据
  uv run python generate_data.py --train-per-topic 4   # 先用很少的数量跑一遍，确认流程没问题
"""
import argparse
import json
import random
from pathlib import Path

from core.llm_gen import build_examples
from core.schema import LABELS
from domains import hotel, townmind, tutor

DATA_DIR = Path(__file__).parent / "data"
UNSAFE_ANGLES = ("对话不耐烦、语气冲", "被反复追问后不耐烦地回怼")


def _gen_for_spec(spec, per_topic: int) -> list[dict]:
    rows = []
    rows += build_examples(spec, "ok", spec.ok_topics, per_topic)
    rows += build_examples(spec, "fabricated", spec.fabrication_topics, per_topic)
    rows += build_examples(spec, "out_of_character", spec.ooc_angles, per_topic)
    rows += build_examples(spec, "unsafe", UNSAFE_ANGLES, per_topic)
    return rows


def _dedup(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out = []
    for r in rows:
        key = (r["domain"], r["reply"])
        if r["reply"] and len(r["reply"]) <= 100 and key not in seen:
            seen.add(key)
            out.append(r)
    return out


_PREFIX = ""


def _shingles(text: str, k: int = 6) -> set[str]:
    return {text[i:i + k] for i in range(max(0, len(text) - k + 1))}


def _drop_overlapping(rows: list[dict], exclude_path: Path) -> list[dict]:
    """留出集纪律：训练句子跟测试集任何一句有 6 字以上连续重合就丢。出题模型和造训练数据的
    模型看的是同一份设定，很容易写出几乎一样的句子；不过滤，测试集就不是留出的。"""
    import json as _json

    data = _json.loads(exclude_path.read_text(encoding="utf-8"))
    test = [_shingles(x["reply"]) for x in data["items"] if x.get("reply")]
    kept, dropped = [], 0
    for r in rows:
        s = _shingles(r["reply"])
        if any(s & t for t in test):
            dropped += 1
            continue
        kept.append(r)
    print(f"  留出集过滤：跟 {exclude_path.name} 有 6 字重合的训练句丢了 {dropped} 条")
    return kept


def _write(name: str, rows: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    name = _PREFIX + name
    path = DATA_DIR / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": f"{name}-{i}", **r}, ensure_ascii=False) + "\n")
    counts = {lb: sum(1 for r in rows if r["label"] == lb) for lb in LABELS}
    print(f"  {name}: {len(rows)} 条  {counts}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-per-topic", type=int, default=12, help="训练集：每个 主题×标签 组合生成几条")
    ap.add_argument("--test-per-topic", type=int, default=4, help="测试集：每个 主题×标签 组合生成几条（不需要很多）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prefix", default="", help="输出文件名前缀，比如 v2_ -> data/v2_train.jsonl；留空就覆盖原文件")
    ap.add_argument("--train-only", action="store_true", help="只生成训练/验证集，不动两份留出测试集")
    ap.add_argument("--exclude", type=Path, default=None,
                    help="一个场景库 JSON（server/evals/scenarios/guard.json）：训练句子跟它里面任何一句"
                         "有 6 字以上重合就丢掉，保证那份测试集是干净的留出集")
    args = ap.parse_args()
    random.seed(args.seed)
    global _PREFIX
    _PREFIX = args.prefix

    print("[1/4] 生成训练数据来源：TownMind + 云雀酒店 ...")
    train_pool: list[dict] = []
    for spec in townmind.specs() + hotel.specs():
        train_pool += _gen_for_spec(spec, args.train_per_topic)
    train_pool = _dedup(train_pool)
    if args.exclude:
        train_pool = _drop_overlapping(train_pool, args.exclude)
    random.shuffle(train_pool)
    cut = max(1, int(len(train_pool) * 0.85))
    _write("train", train_pool[:cut])
    _write("val", train_pool[cut:])

    if args.train_only:
        print("[2/4] --train-only：两份留出测试集保持不动。")
        return
    print("[2/4] 生成同领域留出测试集：TownMind，但用训练时没见过的编造主题 ...")
    indomain: list[dict] = []
    for spec in townmind.specs():
        indomain += build_examples(spec, "fabricated", townmind.TEST_INDOMAIN_FABRICATION_TOPICS, args.test_per_topic)
        # "ok" 也覆盖这些新话题：被问到时应当礼貌拒绝，而不是编——这里复用 ok 的生成指令，
        # 把"没见过的话题"当作 ok 指令里第 (c) 种情况的主题
        indomain += build_examples(spec, "ok", townmind.TEST_INDOMAIN_FABRICATION_TOPICS, args.test_per_topic)
        indomain += build_examples(spec, "ok", spec.ok_topics, args.test_per_topic)
    _write("test_indomain", _dedup(indomain))

    print("[3/4] 生成跨领域测试集：启明在线学习助教——训练数据里一条都不会用到这个领域 ...")
    crossdomain: list[dict] = []
    for spec in tutor.specs():
        crossdomain += _gen_for_spec(spec, args.test_per_topic)
    _write("test_crossdomain", _dedup(crossdomain))

    print("[4/4] 完成。数据在 guard/data/ 下，可以打开看看质量，也可以人工抽查几条标签对不对。")


if __name__ == "__main__":
    main()
