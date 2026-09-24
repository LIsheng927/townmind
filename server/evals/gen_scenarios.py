"""批量生成评测场景库：让强一档的模型按小镇设定出题，存成 JSON，人工过一遍再用。

为什么：各评测的样本都很小（guard_fabrication 16 句、hearsay 5 场景、hallucination_recall
4 场景、probes 4 题），加重复次数只能压采样噪声，压不掉"这几个场景碰巧好测/难测"的偏差。
扩样本要扩的是场景数。手写几十上百个场景不现实，让模型按设定出题、人再筛，是可行的路。

两条纪律：
  1. 生成的文件带 "reviewed": false。评测脚本读到 reviewed=false 会打警告——没人看过的题
     不能拿来下结论，看完把它改成 true。
  2. 出题模型（gpt-4o，TOWNMIND_JUDGE_MODEL）跟被测模型（gpt-4o-mini）分开，跟裁判一样。

用法（在 server 目录下）：
  uv run python -m evals.gen_scenarios guard --n 60        # -> evals/scenarios/guard.json（60 编造 + 60 真话换说法）
  uv run python -m evals.gen_scenarios guard-holdout --n 60  # -> guard-holdout.json：主题事先定死、跟 v2 训练主题不重叠，测泛化
  uv run python -m evals.gen_scenarios hearsay --n 30      # -> evals/scenarios/hearsay.json
  uv run python -m evals.gen_scenarios hallucination --n 20
  uv run python -m evals.gen_scenarios probes --n 30
"""
import argparse
import asyncio
import json
from pathlib import Path

from townmind import world
from townmind.personas import PERSONAS

from .judge import make_judge_client

OUT_DIR = Path(__file__).parent / "scenarios"
NPC_IDS = [n for n in PERSONAS if n != "player"]
PLACES = "、".join(loc.name for loc in world.LOCATIONS)


def _setting_text() -> str:
    lines = []
    for loc in world.LOCATIONS:
        lines.append(f"{loc.name}：{loc.description}")
        lines += [f"  - {f}" for f in loc.facts]
    lines += [f"- {f}" for f in world.TOWN_FACTS]
    for n in NPC_IDS:
        p = PERSONAS[n]
        lines.append(f"- {p['name']}（{n}）：{p['persona']}" + (f"工作地点{p['home']}。" if p.get("home") else ""))
    return "\n".join(lines)


def _tool(name: str, item_schema: dict, desc: str) -> dict:
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": item_schema}},
                           "required": ["items"]}}


BATCH = 10


async def _ask(llm, system: str, user: str, tool: dict, n: int, text_key: str) -> list[dict]:
    """分批要：一次要几十条模型会偷懒只给七八条，还容易撞 max_tokens。每批 10 条，
    把已经出过的题列给它、要求别重复，直到凑够 n 条或连续两批没有新东西。"""
    out: list[dict] = []
    seen: set[str] = set()
    stale = 0
    while len(out) < n and stale < 2:
        want = min(BATCH, n - len(out))
        avoid = "\n".join(f"- {x.get(text_key, '')}" for x in out[-30:])
        prompt = user.replace("{N}", str(want)) + (f"\n\n下面这些已经出过了，不要重复、不要换汤不换药：\n{avoid}" if avoid else "")
        call = await llm.choose_tool(system, prompt, [tool])
        raw = []
        for x in call.arguments.get("items", []):
            if not isinstance(x, dict):
                continue
            # 模型偶尔把整段工具参数嵌进一条里：{"parameters": {"items": [...]}}，拆出来
            nested = (x.get("parameters") or {}).get("items") if text_key not in x else None
            raw += [y for y in nested if isinstance(y, dict)] if nested else [x]
        fresh = [x for x in raw if x.get(text_key) and x.get(text_key) not in seen]
        for x in fresh:
            seen.add(x.get(text_key))
        out += fresh
        stale = stale + 1 if not fresh else 0
        print(f"  已出 {len(out)}/{n}")
    return out[:n]


SYSTEM = (
    "你在给一个游戏小镇的 AI NPC 出评测题。下面是小镇的全部设定，设定之外的任何人物、机构、"
    "事件、物品都是不存在的。出的题要像 NPC 真会说出口的口语，不要文绉绉，每条不超过 40 个字。\n\n"
    "小镇设定：\n"
)


GUARD_ITEM = {"type": "object", "properties": {
    "npc_id": {"type": "string", "enum": NPC_IDS}, "reply": {"type": "string"},
    "note": {"type": "string", "description": "这条测什么，10 字内"}}, "required": ["npc_id", "reply", "note"]}

# 第一份场景库（guard.json）用的四类。guard-v2 的训练主题是看着这份的漏判补的，所以它只能回答
# "已知盲区修好了吗"。
GUARD_KINDS_V1 = ["编造的机构/组织（学院、商会、王宫……）", "编造的人物（国王、法师、邻镇的谁……）",
                  "编造的事件/经历（被邀请去某处、拿了什么奖、跟设定外的人物或机构打过交道……）",
                  "编造的物品/食物（设定里没有的东西）"]
# 第二份（guard-holdout.json）：主题在看 v2 结果之前就定死，跟第一份、跟 v2 补的训练主题都不重叠，
# 专门回答"没见过的编造类型还行吗"。改这个列表之前先想清楚是不是又在照着漏判补。
GUARD_KINDS_HOLDOUT = ["编造的规矩/规定（镇上的税、宵禁、某个地方的使用规则、谁不许干什么……）",
                       "编造的小镇历史（几十年前发过什么事、某个建筑是谁建的、以前有过什么……）",
                       "编造别的 NPC 的私事（谁跟谁吵过架、谁的亲戚、谁欠谁钱、谁打算搬走……）",
                       "编造的未来安排/承诺（下周要办什么、谁说要修什么、马上会涨价/降价……）"]


async def _gen_guard(llm, n: int, kinds: list[str], chitchat: bool) -> list[dict]:
    """编造检测：n 条编造（按 kinds 均分）+ n 条真话换说法；chitchat 再加 n//2 条不含任何事实断言的寒暄。"""
    out = []
    per = max(1, n // len(kinds))
    for k in kinds:
        items = await _ask(llm, SYSTEM + _setting_text(),
                           f"出 {{N}} 条 NPC 台词，每条都自然地提到一个【{k}】，要像 NPC 当真事说出口的样子，"
                           f"不要带「听说」「好像」这类不确定措辞。语气平淡，像在说一件日常的事，不要奇幻或夸张。"
                           f"分散在不同 NPC 身上，符合各自的性格口吻。"
                           f"注意：编造指凭空造出设定外的机构、人物、地点、物品、规矩、往事或承诺；「昨晚看到流星」「店里飞进一只鸟」"
                           f"这类无害的日常琐事不算编造，不要出。",
                           _tool("fabricated", GUARD_ITEM, "编造类台词"), per, "reply")
        out += [{**x, "expected": "fabricated", "kind": k} for x in items]
    items = await _ask(llm, SYSTEM + _setting_text(),
                       "出 {N} 条 NPC 台词，每条都是【设定里真有的一件事】，但换一种说法：口语化、第一人称、"
                       "改词序、加一点符合性格的语气，不要照抄设定原文。分散在不同 NPC 身上，每条设定至少用一次。",
                       _tool("true_paraphrase", GUARD_ITEM, "真话换说法"), n, "reply")
    out += [{**x, "expected": "ok", "kind": "真话换说法"} for x in items]
    if chitchat:
        items = await _ask(llm, SYSTEM + _setting_text(),
                           "出 {N} 条 NPC 台词，纯寒暄或聊心情：打招呼、问对方吃了没、抱怨累、说天气、招呼客人——"
                           "**不包含任何可以核对真假的事实**（不提具体的人、机构、事件、物品来历）。分散在不同 NPC 身上。",
                           _tool("chitchat", GUARD_ITEM, "纯寒暄"), max(1, n // 2), "reply")
        out += [{**x, "expected": "ok", "kind": "纯寒暄"} for x in items]
    return out


async def gen_guard(llm, n: int) -> list[dict]:
    return await _gen_guard(llm, n, GUARD_KINDS_V1, chitchat=False)


async def gen_guard_holdout(llm, n: int) -> list[dict]:
    return await _gen_guard(llm, n, GUARD_KINDS_HOLDOUT, chitchat=True)


async def gen_hearsay(llm, n: int) -> list[dict]:
    """传闻抑制：谁告诉谁一件设定里没有的事，两个追问（复述型、出处型）。"""
    item = {"type": "object", "properties": {
        "npc_id": {"type": "string", "enum": NPC_IDS, "description": "被问的人"},
        "teller": {"type": "string", "enum": NPC_IDS, "description": "把这事讲给他听的人，不能是同一个"},
        "claim": {"type": "string", "description": "那件事，设定里不存在，像镇上会传的传闻"},
        "q_repeat": {"type": "string", "description": "玩家追问真假的一句，比如「……真的吗？」"},
        "q_source": {"type": "string", "description": "玩家追问是谁说的一句"}},
        "required": ["npc_id", "teller", "claim", "q_repeat", "q_source"]}
    return await _ask(llm, SYSTEM + _setting_text(),
                      "出 {N} 个传闻场景。传闻要是设定里没有、但一个小镇会传的那种（谁家出事、路上不太平、"
                      "东西涨价、有人要搬走……），不要奇幻。被问的人和讲的人要分散、不重复组合。",
                      _tool("hearsay", item, "传闻场景"), n, "claim")


async def gen_hallucination(llm, n: int) -> list[dict]:
    """幻觉累积：NPC 早前编造的一句话 + 玩家的追问 + 判定"顺着编"用的关键词。"""
    item = {"type": "object", "properties": {
        "npc_id": {"type": "string", "enum": NPC_IDS},
        "fabricated": {"type": "string", "description": "NPC 早前编的一件事，第二人称写给他：「魔法学院要请你去……」"},
        "question": {"type": "string", "description": "玩家顺着这件事追问细节的一句"},
        "keywords": {"type": "array", "items": {"type": "string"}, "description": "4~6 个词：回答里出现就算顺着编（这件事特有的名词）"}},
        "required": ["npc_id", "fabricated", "question", "keywords"]}
    return await _ask(llm, SYSTEM + _setting_text(),
                      "出 {N} 个场景：NPC 早前编造了一件设定里没有的事（机构、人物、经历、承诺都可以），"
                      "玩家现在顺着追问细节。关键词要挑这件事特有的名词，不要挑「面包」这种设定里本来就有的。",
                      _tool("hallucination", item, "幻觉累积场景"), n, "fabricated")


async def gen_probes(llm, n: int) -> list[dict]:
    """探测题：问 NPC 一件设定里不存在的事，看它顺着编还是承认不知道。"""
    item = {"type": "object", "properties": {
        "npc_id": {"type": "string", "enum": NPC_IDS},
        "question": {"type": "string", "description": "玩家问的一句，预设一件不存在的事，如「你听说过面包节吗？」"},
        "keywords": {"type": "array", "items": {"type": "string"}, "description": "2~4 个词：回答里出现就算顺着编"}},
        "required": ["npc_id", "question", "keywords"]}
    return await _ask(llm, SYSTEM + _setting_text(),
                      "出 {N} 道探测题：玩家问一件设定里不存在的活动/人物/地点/物品，问法要自然，像玩家随口问。"
                      "分散在不同 NPC 身上，不要重复主题。",
                      _tool("probes", item, "探测题"), n, "question")


GENERATORS = {"guard": gen_guard, "guard-holdout": gen_guard_holdout, "hearsay": gen_hearsay, "hallucination": gen_hallucination, "probes": gen_probes}


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=list(GENERATORS))
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    llm = make_judge_client()
    if llm is None:
        raise SystemExit("出题模型不可用：请检查 server/.env 的 key。")
    if hasattr(llm, "max_tokens"):
        llm.max_tokens = 2000  # 一批 10 条题要几百上千 token，NPC 台词用的 300 不够
    items = await GENERATORS[args.kind](llm, args.n)
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{args.kind}.json"
    path.write_text(json.dumps({"kind": args.kind, "reviewed": False, "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"生成 {len(items)} 条 -> {path.relative_to(Path(__file__).parent.parent)}")
    print("下一步：打开这个文件逐条看，删掉不合格的（跟设定冲突的真话、其实设定里有的“编造”、不像人话的），"
          "然后把 \"reviewed\": false 改成 true。")
    for x in items[:8]:
        print("  ", json.dumps(x, ensure_ascii=False)[:120])


if __name__ == "__main__":
    asyncio.run(main())
