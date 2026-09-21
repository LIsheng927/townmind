"""领域无关的数据生成引擎：给一个 WorldSpec + 一个标签 + 一个主题，
调用真实大模型批量造出"这个角色会怎么说"的台词，全部脱敏成 (persona, context, reply, label) 这种通用格式。
"""
import json
import re

from .llm import generate_text
from .schema import WorldSpec

_SYSTEM = (
    "你在帮助制作一个内容审核模型的训练数据。你要扮演下面这个虚构角色，"
    "按要求写出一批它可能会说的话。只输出一个 JSON 数组，元素是字符串，不要任何其他文字。"
)

_INSTRUCTIONS = {
    "ok": (
        "写 {n} 句这个角色会说的话，每句必须满足下面三种情况之一，三种都要有：\n"
        "  (a) 完全符合背景资料的内容；\n"
        "  (b) 和背景资料无关的正常寒暄/闲聊（问候、聊心情、聊天气等）；\n"
        "  (c) 角色被问到背景资料里没有的事（关于「{topic}」），礼貌地表示不知道/不清楚/拒绝，"
        "但没有编造任何细节。\n"
        "每句独立、话题和句式都不同，长度不超过 40 字，符合角色设定的语气。"
    ),
    "fabricated": (
        "写 {n} 句这个角色说的话，每句都要编造一件关于「{topic}」、但背景资料里完全没有提到的具体事情"
        "（必须包含编出来的具体细节，比如名字、数量、时间、价格，不能只是含糊带过）。"
        "注意：单纯说「没有/不提供/不清楚」这种如实拒绝、没有编出任何新细节的句子不算编造，不要写这种。"
        "每句独立、说法不同，长度不超过 40 字，符合角色设定的语气。"
    ),
    "out_of_character": (
        "写 {n} 句这个角色说的话，每句都要违背角色设定，具体角度是「{topic}」。"
        "每句独立、说法不同，长度不超过 60 字。"
    ),
    "unsafe": (
        "这些例子会被用来训练一个安全分类器，帮它学会识别「语气恶劣、对提问者不礼貌」的回复，"
        "从而在真实产品里自动拦截，避免这种回复真的发给用户（包括未成年学生这样的场景）。"
        "写 {n} 句这个角色态度差、明显不耐烦、敷衍甚至阴阳怪气的话（比如嫌对方烦、不耐烦地打发对方），"
        "不要写脏话、人身攻击、威胁或真实姓名，体现「态度差/不耐烦」这种程度即可。"
        "每句独立、说法不同，长度不超过 30 字。"
    ),
}


def _extract_json_array(text: str) -> list[str]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(x).strip() for x in data if isinstance(x, (str, int, float)) and str(x).strip()]


def generate_batch(spec: WorldSpec, label: str, topic: str, n: int = 6, temperature: float = 1.1) -> list[str]:
    """造一批某个标签、某个主题的台词。网络/解析失败时返回空列表，由调用方决定要不要重试或跳过。"""
    user = (
        f"【角色设定】{spec.persona}\n"
        f"【背景资料】\n{spec.context}\n\n"
        + _INSTRUCTIONS[label].format(n=n, topic=topic)
    )
    try:
        raw = generate_text(_SYSTEM, user, temperature=temperature)
    except Exception as e:  # 网络抖动/限流：跳过这一批，不让整个生成流程崩掉
        print(f"  ! 生成失败（{spec.domain}/{label}/{topic}）：{type(e).__name__}: {e}")
        return []
    items = _extract_json_array(raw)
    if not items and raw.strip():
        # 模型没报错，但也没输出能解析的 JSON 数组——多半是拒绝写了/换了格式，打出来方便排查
        preview = raw.strip().replace("\n", " ")[:120]
        print(f"  ? 解析出 0 条（{spec.domain}/{label}/{topic}），原始输出开头：{preview!r}")
    return items


_BARE_DENIAL = re.compile(r"不提供|没有开设|不开设|无法学习|没有这|不存在|没有提供|并没有|暂不支持|不支持|没有相关|不能提供|没有安排")
_INVENTED_DETAIL = re.compile(
    r"[0-9\uff10-\uff19]|[二三四五六七八九十两百千万]|"
    r"[「『\u201c\u2018][^」』\u201d\u2019]{1,12}[」』\u201d\u2019]"
)


def _is_bare_denial(label: str, text: str) -> bool:
    """训练数据质量过滤：fabricated 标签下，如果一句话只是如实说"没有/不提供"、
    没有编出任何具体细节（数字/量词/带引号的自造名词），说明生成模型偷懒写成了
    「如实拒绝」而不是「编造」，这种要丢掉，否则模型会学错「拒绝 vs 编造」的边界。"""
    if label != "fabricated":
        return False
    return bool(_BARE_DENIAL.search(text)) and not _INVENTED_DETAIL.search(text)


def build_examples(spec: WorldSpec, label: str, topics: tuple[str, ...], per_topic: int = 6) -> list[dict]:
    """对一个标签下的每个主题各生成一批，拼成统一格式的样本列表。"""
    out: list[dict] = []
    for topic in topics:
        for reply in generate_batch(spec, label, topic, n=per_topic):
            if _is_bare_denial(label, reply):
                continue
            out.append(
                {"domain": spec.domain, "persona": spec.persona, "context": spec.context,
                 "label": label, "topic": topic, "reply": reply}  # fmt: skip
            )
    return out
