"""多层安全的"检查员"：纯函数，不联网，可以单独测试。

  入口检查 check_player_text：玩家说的话进入系统之前先过一遍（挡住过长、注入、越狱、破坏提示词格式）
  出口检查 check_npc_reply ：大模型说完话，发出去之前再过一遍（挡住出戏、泄露提示词、太长、编造设定里没有的东西）

所有规则都是启发式的（关键词 + 正则），会有漏判和误判：它们是"多层防线里的一层"，不是万能锁。
每一层的效果，靠 evals/attacks.py 里的攻击题量化。
"""
import re
from dataclasses import dataclass, field

from . import world

MAX_PLAYER_CHARS = 100  # 玩家一句话最长多少字（超出的部分直接截掉）
MAX_NPC_CHARS = 60  # 提示词要求 NPC 说话不超过 30 字；这是硬上限，给一点余量

# ---------------- 入口：玩家输入 ----------------
_INJECTION = re.compile(
    r"(忽略|无视|忘掉|忘记|不要管|不用管).{0,8}(之前|以上|上面|前面|所有|你的).{0,8}(指令|设定|规则|提示|要求|限制)"
    r"|(你现在|从现在起|从现在开始).{0,4}(是|扮演|变成)"
    r"|扮演.{0,6}(一个|一名|不受)"
    r"|(system\s*prompt|系统提示|提示词|初始设定|你的设定|你的指令)"
    r"|(输出|重复|复述|告诉我|说出).{0,6}(上面|以上|你的).{0,6}(内容|指令|设定|提示|规则)"
    r"|(开发者模式|越狱|jailbreak|\bDAN\b|ignore (all|previous|the above))",
    re.IGNORECASE,
)
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_QUOTE_MAP = str.maketrans({"「": "『", "」": "』"})  # 提示词用「」包住对话：玩家不能用它们"跳出"引号


@dataclass
class GuardResult:
    ok: bool  # False = 这句话应当被拦下/替换
    text: str  # 处理后的文本（入口：清洗后的；出口：原文）
    flags: list[str] = field(default_factory=list)  # 触发了哪些规则，用于统计和调试


def check_player_text(text: str) -> GuardResult:
    """入口检查。结果：ok=False 表示为空，直接丢弃；flags 非空表示"可疑"，仍会放行但要打标记，
    并让 NPC 用角色身份婉拒（见 agent 的提示词）。"""
    flags: list[str] = []
    cleaned = _CONTROL.sub("", text or "").strip()
    if not cleaned:
        return GuardResult(False, "", ["empty"])
    if len(cleaned) > MAX_PLAYER_CHARS:
        cleaned = cleaned[:MAX_PLAYER_CHARS]
        flags.append("truncated")
    if "「" in cleaned or "」" in cleaned:
        cleaned = cleaned.translate(_QUOTE_MAP)
        flags.append("quote_escape")
    if _INJECTION.search(cleaned):
        flags.append("injection")
    return GuardResult(True, cleaned, flags)


# ---------------- 出口：NPC 说的话 ----------------
_LEAK_OR_OOC = re.compile(
    r"(系统提示|提示词|system\s*prompt|我的设定是|我被设定|我的指令|工具调用|function\s*call)"
    r"|作为(一个|一名)?\s*(AI|人工智能|语言模型|大模型|助手)"
    r"|我是(一个|一名)?\s*(AI|人工智能|语言模型|大模型|聊天机器人)"
    r"|(OpenAI|ChatGPT|GPT-?\d|Anthropic)",
    re.IGNORECASE,
)

# 设定里允许提到的食物/物品（要和 world.py 里的描述保持一致，tests 里有一致性检查）
ALLOWED_ITEMS = ("蓝莓松饼", "肉桂卷", "法棍", "松饼", "面包", "面粉", "铁矿", "矿石", "铁器", "农具", "刀具")
# 常见的食物/物品词：说到了这些，又不在设定里，就算编造
_ITEM_CORE = re.compile(
    r"(?:提拉米苏|蛋糕|饼干|饼|糕|卷|派|塔|酥|披萨|咖啡|奶茶|茶|酒|汤|面条|米饭|"
    r"巧克力|布丁|甜甜圈|马卡龙|可颂|吐司|三明治|汉堡)"
)


def ungrounded_items(text: str) -> list[str]:
    """NPC 提到了设定里没有的食物/物品（例如"香草提拉米苏"里的"提拉米苏"、"巧克力法棍"里的"巧克力"）。

    做法：先把设定里允许的词整个抠掉，再在剩下的文字里找常见食物词；剩下的就是"没有依据的"。
    """
    rest = text
    for w in sorted(ALLOWED_ITEMS, key=len, reverse=True):
        rest = rest.replace(w, "\u0000")  # 用占位符隔开，避免拼出新词
    return [m.group() for m in _ITEM_CORE.finditer(rest)]


_DENIAL = re.compile(r"没听说|没听过|不知道|不清楚|不了解|没有这|没这|不存在|没见过|不认识|镇上没有|这里没有")


def check_npc_reply(text: str, heard_text: str = "") -> GuardResult:
    """出口检查。ok=False：这句话不该发出去，应当换成行为树的台词，并且不写进记忆。

    heard_text：NPC 刚听到的话。如果对方提到了设定里没有的东西，NPC 说"没听说过提拉米苏"这样的否认，
    复述这个词是合理的，不应被拦；但顺着说"提拉米苏很好吃"仍然会被拦。
    """
    flags: list[str] = []
    if not text or not text.strip():
        return GuardResult(False, text, ["empty"])
    if len(text) > MAX_NPC_CHARS:
        flags.append("too_long")
    if _LEAK_OR_OOC.search(text):
        flags.append("leak_or_out_of_character")
    bad = ungrounded_items(text)
    if bad and _DENIAL.search(text):
        echoed = set(ungrounded_items(heard_text))
        bad = [b for b in bad if b not in echoed]
    if bad:
        flags.append("ungrounded_item")
    return GuardResult(not flags, text, flags)
