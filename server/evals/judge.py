"""LLM-as-judge：用一个（更强的）大模型给评测里的回复打标签，替代/校验关键词判定。

为什么需要：README 里至少三次记录了关键词判定漏掉某种说法、补上之后数字变了
（reflection_quality 的 grounded_rate、dialogue_style 的 FILLERS、hallucination_recall 的
CORRECTION）。关键词只能量化"有没有说出特定措辞"，不是真的理解这句话在做什么——
"具体到账时间我没听说过呢"是在否认，却因为带了"到账"两个字被判成"顺着编"。
RAG/对话评测这两年的标准做法是让一个强模型当裁判，给出标签 + 一句理由。

怎么用：各评测脚本加 --judge 之后，每条回复除了关键词判定，再让裁判判一次；结果表
两套数字并列，外加一致率和 Cohen's κ——裁判不是"更对"的真理，两者不一致的那些
才是该人工去看的（结果 md 里单独列出来）。

裁判模型跟被测的 NPC 模型分开配（TOWNMIND_JUDGE_MODEL），默认用比 NPC 强一档的：
NPC 是 gpt-4o-mini，裁判默认 gpt-4o。用同一个模型自己判自己，偏差会往同一个方向倒。
输出走 tool calling（label 是枚举），跟项目里其它地方一样不解析自由文本。
"""
import asyncio
import os
from collections import Counter

from townmind.llm.base import LLMClient, ToolCall

ENV_JUDGE_MODEL = "TOWNMIND_JUDGE_MODEL"
DEFAULT_JUDGE_MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-4-5"}


def make_judge_client() -> LLMClient | None:
    """跟 townmind.llm.factory.make_client 同一套 provider/key，只是换成裁判模型。"""
    from dotenv import load_dotenv

    from townmind.llm.factory import ENV_PATH

    load_dotenv(ENV_PATH)
    provider = os.getenv("TOWNMIND_LLM_PROVIDER", "anthropic").lower()
    model = os.getenv(ENV_JUDGE_MODEL) or DEFAULT_JUDGE_MODELS.get(provider)
    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        from townmind.llm.anthropic_client import AnthropicClient

        return AnthropicClient(model)
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        from townmind.llm.openai_client import OpenAIClient

        return OpenAIClient(model)
    return None


class Judge:
    def __init__(self, llm: LLMClient, max_concurrent: int = 4) -> None:
        self.llm = llm
        self._slots = asyncio.Semaphore(max_concurrent)
        self.calls = 0
        self.failures = 0

    async def label(self, task: str, context: str, reply: str, labels: dict[str, str]) -> tuple[str, str] | None:
        """给一条回复打标签。labels = {标签: 这个标签的定义}。返回 (标签, 一句理由)；
        调用失败或模型给了不在名单里的标签，返回 None（调用方按"裁判缺席"处理，不当成某个标签）。"""
        system = (
            "你是游戏对话评测的裁判。下面会给你一段情境和 NPC 的一句回复，请严格按定义选一个标签，"
            "并用一句话说明理由。只看这句回复本身在做什么，不要脑补它的意图。"
        )
        defs = "\n".join(f"- {k}：{v}" for k, v in labels.items())
        user = f"任务：{task}\n\n情境：\n{context}\n\nNPC 的回复：「{reply}」\n\n标签定义：\n{defs}"
        tool = {
            "name": "label",
            "description": "给这条回复打标签",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": list(labels)},
                    "reason": {"type": "string", "description": "一句话理由，不超过 40 字"},
                },
                "required": ["label", "reason"],
            },
        }
        self.calls += 1
        try:
            async with self._slots:
                call: ToolCall = await asyncio.wait_for(self.llm.choose_tool(system, user, [tool]), 30)
        except Exception:
            self.failures += 1
            return None
        lab = call.arguments.get("label")
        if lab not in labels:
            self.failures += 1
            return None
        return lab, str(call.arguments.get("reason", ""))


def agreement(pairs: list[tuple[str, str]]) -> dict:
    """两套标签的一致率和 Cohen's κ。pairs = [(关键词标签, 裁判标签), ...]，裁判缺席的不算。

    κ 把"碰巧一致"扣掉：两套标签都几乎总是打同一个值时，一致率会虚高，κ 不会。
    读法（Landis & Koch 的惯例）：< 0.2 几乎不一致，0.4~0.6 中等，> 0.8 几乎完全一致。"""
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None}
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "agreement": po, "kappa": kappa}


def render_agreement(stats: dict, what: str) -> str:
    if not stats["n"]:
        return f"{what}：裁判没有可用结果。"
    return f"{what}：关键词 vs 裁判一致率 {stats['agreement']:.0%}，Cohen's κ = {stats['kappa']:.2f}（n={stats['n']}）"


def render_disagreements(rows: list[dict], key_kw: str, key_judge: str, key_reason: str, key_text: str, limit: int = 30) -> str:
    out = []
    for r in rows:
        if r.get(key_judge) is None or r[key_kw] == r[key_judge]:
            continue
        out.append(f"关键词={r[key_kw]:<12} 裁判={r[key_judge]:<12} 「{r[key_text]}」\n        裁判理由：{r.get(key_reason, '')}")
        if len(out) >= limit:
            break
    return "\n".join(out) if out else "（两套判定完全一致）"
