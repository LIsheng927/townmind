"""带依据的事实核查（NLI 蕴含）：一句话有没有被"设定 + 这个 NPC 的记忆"支持。

为什么在 guard 分类器之外还要这一层：guard（guard_model.py）是"看一句话，判它是不是
编造"，它没有拿着依据去对——训练数据里当时把"镇长"当成不存在的人物，后来设定里加了
"镇长会巡视广场"，它就可能把一句真话判成编造（evals/guard_fabrication.py 专门放了这条
反例）。而且它判错的后果比编造更糟：结构性修复那条链路会拿 guard 的结论去让 NPC
"承认自己说错了"，判错就是让 NPC 否认一件真事。

这一层换个问法，是 MiniCheck / RAGAS faithfulness 这类 2024~2026 年主流做法的形式：
给两段文本——一份依据（premise）和一句陈述（hypothesis），问"依据能不能推出这句话"。
依据里有镇长它就不会判假，设定改了它自动跟上，不需要重新训练。

但它不能单独当出口检查用，这一点要先说清楚：NPC 的台词大多是寒暄（"要不要来块面包"），
跟设定的关系是"中立"而不是"蕴含"，单看蕴含分数会把所有寒暄都当成没依据。所以它在
这个项目里的角色是**证据否决**：只在 guard 说"编造"的时候出面，如果依据明明能推出这句话，
就推翻 guard 的判断。这样只用到 NLI 最擅长的一半——"有依据"这个方向的高精度。

MiniCheck 系列模型只有英文；这里用的是同一思路的多语言 NLI 模型（XNLI 覆盖中文）：
MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7，约 2.8 亿参数，GPU 上
一批二三十条依据一次前向几十毫秒。

跟 guard_model.py 一样是彻底可选的：没装 torch/transformers、没下到模型、推理异常，
都当作"这一层不可用"，返回 None，绝不拖垮主流程。推理是同步阻塞的，调用方要
asyncio.to_thread。
"""
import logging
import os

from . import world
from .personas import DEFAULT_PERSONA, PERSONAS

log = logging.getLogger("townmind.grounding")

ENV_ENABLE = "TOWNMIND_USE_GROUNDING"
ENV_MODEL = "TOWNMIND_GROUNDING_MODEL"
DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
# 蕴含概率到这个数以上，才算"依据确实支持这句话"、有资格推翻 guard。定得偏高是有意的：
# 这一层只负责救回被误判的真话，宁可少救也不能把真编造放过去
VETO_THRESHOLD = 0.7


def evidence_for(npc_id: str, memories: list[str] = ()) -> list[str]:
    """一个 NPC 此刻"有依据"的全部素材：地点设定、镇上的事、自己的人设、以及传进来的记忆。
    每条单独作为一个 premise，跟陈述配对——不拼成一大段，是因为 NLI 模型是在句对上训的，
    一段几十行的 premise 里真正相关的那句会被稀释。"""
    lines: list[str] = []
    for loc in world.LOCATIONS:
        lines.append(f"{loc.name}：{loc.description}")
        lines.extend(loc.facts)
    lines.extend(world.TOWN_FACTS)
    p = PERSONAS.get(npc_id, DEFAULT_PERSONA)
    if p.get("persona"):
        lines.append(p["persona"])
    if p.get("home"):
        lines.append(f"{p['name']}的工作地点是{p['home']}。")
    lines.extend(m for m in memories if m)
    return lines


class GroundingChecker:
    """惰性加载，跟 GuardModel 一个套路：构造时什么都不做，第一次调用才加载；失败就记一次
    日志，之后直接短路返回 None。"""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get(ENV_MODEL) or DEFAULT_MODEL
        self._model = None
        self._tokenizer = None
        self._device = None
        self._entail_idx: int | None = None
        self._unavailable = False

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._model is not None:
            return self._model
        if self._unavailable:
            return None
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(self._device).eval()
            labels = {v.lower(): k for k, v in model.config.id2label.items()}
            self._entail_idx = labels["entailment"]
            self._model = model
            log.info("grounding checker 已加载：%s（%s）", self.model_name, self._device)
        except Exception as e:  # 缺依赖 / 没网下不到模型 / 权重损坏：这一层不可用
            log.warning("grounding checker 不可用（%s: %s），这一层跳过", type(e).__name__, e)
            self._unavailable = True
            return None
        return self._model

    def entailment(self, evidence: list[str], statement: str) -> float | None:
        """依据里最能支持这句话的那条，给出的蕴含概率（0~1）；不可用或异常返回 None。"""
        model = self._load()
        if model is None or not evidence or not statement.strip():
            return None
        import torch

        try:
            batch = self._tokenizer(
                evidence, [statement] * len(evidence),
                padding=True, truncation=True, max_length=256, return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                probs = torch.softmax(model(**batch).logits, dim=-1)[:, self._entail_idx]
            return float(probs.max().item())
        except Exception as e:
            log.warning("grounding 推理失败（%s: %s），当作没有这一层", type(e).__name__, e)
            return None

    def supported(self, npc_id: str, statement: str, memories: list[str] = ()) -> float | None:
        """这句话被"设定 + 人设 + 给定记忆"支持的程度。"""
        return self.entailment(evidence_for(npc_id, memories), statement)
