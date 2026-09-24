"""guard/ 训练出来的 LoRA 防御分类器，接进 safety 检查里的一层可选增强。

跟 guard/evaluate.py 里做的事一样（同一套 persona/context/reply -> ok/fabricated/
out_of_character/unsafe 四分类），区别是这里是给正在跑的服务用的，不是离线评测脚本。

这一层是彻底可选的，不是硬依赖：
  - 没装 torch/transformers/peft（server/pyproject.toml 里的 guard-model 可选依赖组，
    默认 `uv sync` 不装），就自动跳过。
  - 没跑过 guard/train.py、adapters/ 目录下没有训练好的权重，也自动跳过。
  - 加载或推理过程中出了任何异常，也当作"这一层不可用"处理，不会把异常抛给调用方。
跟这个项目里"大模型挂了退回行为树""熔断器打开就不发请求"是同一个哲学：可选的增强层，
失败了就当没有这层，服务永远能正常跑，绝不会因为这一层拖垮主流程。

四个标签怎么映射成 safety 的 flag：
  - ok               -> 什么都不加
  - fabricated       -> 跟 safety.check_npc_reply 的 "ungrounded_item" 是同一件事，
                        但正则规则已经能查大部分这类问题了，guard model 是在正则判定
                        "ok" 之后再补一道检查，只会让系统更严格，不会放过正则已经拦下的
  - out_of_character -> 对应 safety 的 "leak_or_out_of_character"，同上，属于补充检查
  - unsafe           -> 正则规则里完全没有"语气差/不耐烦"这类检测（见 guard/evaluate.py
                        里 regex_baseline_predict 的说明），这是 guard model 唯一能查到、
                        正则查不到的新东西，是这一层真正的价值所在

推理是同步、阻塞的（CPU 上跑一次生成有可能要几百毫秒到一两秒），调用方必须用
asyncio.to_thread 之类的方式丢到线程里跑，不能直接在事件循环里同步调用，不然会卡住
所有正在并发处理的 NPC——这正是上一轮并发压测要保护的东西。
"""
import importlib.util
import logging
import os
from pathlib import Path
from types import ModuleType

from . import world
from .personas import DEFAULT_PERSONA, PERSONAS

log = logging.getLogger("townmind.guard_model")

ENV_ENABLE = "TOWNMIND_USE_GUARD_MODEL"  # 设了（任意非空值）才会在 main.py 里构造 GuardModel
ENV_ADAPTER_DIR = "TOWNMIND_GUARD_MODEL_DIR"  # 不设就用下面这个默认路径

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER_DIR = _REPO_ROOT / "guard" / "adapters" / "guard-v2"  # v1 在 111 条大样本上只抓住 51% 的编造，见 README
_GUARD_CORE_DIR = _REPO_ROOT / "guard" / "core"


def _load_module_from_file(name: str, path: Path) -> ModuleType:
    """按文件路径直接加载一个模块，不走 sys.path/包安装那一套。

    guard/core/schema.py（判断用的 prompt 模板、四个标签）和 guard/core/chat.py
    （chat_prompt_ids）本身零第三方依赖，训练、评测、这里三处必须用同一份逻辑——
    prompt 模板对不上，训练就白做了（chat.py 自己的注释也是这么说的）。直接按路径加载
    这两个文件，而不是 sys.path.insert 再 import，是为了不污染 sys.path（"core" 这个
    模块名太通用，容易跟别的东西撞名），也不要求 guard/ 目录在部署时一定存在于
    Python path 上——找不到文件就按"这一层不可用"处理，跟其它失败模式一致。
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _full_world_context() -> str:
    """跟 guard/domains/townmind.py 的 _full_context() 是同一份逻辑（同一份 world 数据，
    这里直接从 townmind.world 现算，不需要跨项目导入）：训练时喂的 context 是这个格式，
    推理时也必须是这个格式，否则就是拿模型没见过的输入格式去问它，准确率无从谈起。"""
    lines: list[str] = []
    for loc in world.LOCATIONS:
        lines.append(f"{loc.name}：{loc.description}")
        lines += [f"  - {f}" for f in loc.facts]
    lines += list(world.TOWN_FACTS)
    return "\n".join(lines)


def _persona_for(npc_id: str) -> str:
    """跟 guard/domains/townmind.py 的 _persona_for() 是同一个拼法，训练数据就是这么拼的。"""
    p = PERSONAS.get(npc_id, DEFAULT_PERSONA)
    home = f"你的工作地点是{p['home']}。" if p.get("home") else ""
    return f"你是游戏小镇里的 NPC「{p['name']}」。{p['persona']}{home}"


class GuardModel:
    """惰性加载：构造时什么都不做，第一次真正调用 classify() 才尝试加载模型；
    加载失败（缺依赖、没有训练好的权重、权重损坏……）就记一次日志，之后的调用
    直接短路返回 None，不会每次都重新尝试加载。"""

    def __init__(self, adapter_dir: Path | str | None = None) -> None:
        self.adapter_dir = Path(adapter_dir or os.environ.get(ENV_ADAPTER_DIR) or DEFAULT_ADAPTER_DIR)
        self._model = None
        self._tokenizer = None
        self._device = None
        self._unavailable = False
        self._labels: tuple[str, ...] | None = None
        self._build_judge_prompt = None
        self._chat_prompt_ids = None

    @property
    def available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._model is not None:
            return self._model
        if self._unavailable:
            return None
        try:
            schema = _load_module_from_file("_guard_schema", _GUARD_CORE_DIR / "schema.py")
            chat = _load_module_from_file("_guard_chat", _GUARD_CORE_DIR / "chat.py")
        except (ImportError, OSError) as e:
            log.info("guard model 未启用：找不到 guard/core（%s），这一层保持关闭", e)
            self._unavailable = True
            return None
        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            log.info(
                "guard model 未启用：没装 torch/transformers/peft"
                "（可选依赖，`uv sync --group guard-model` 安装），这一层保持关闭"
            )
            self._unavailable = True
            return None
        if not self.adapter_dir.exists():
            log.info("guard model 未启用：找不到训练好的 adapter（%s），这一层保持关闭", self.adapter_dir)
            self._unavailable = True
            return None
        try:
            peft_config = PeftConfig.from_pretrained(str(self.adapter_dir))
            tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_dir))
            base = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(base, str(self.adapter_dir))
            model.eval()
        except Exception as e:  # 权重损坏/显存不够/版本不兼容……都当成"这一层不可用"
            log.warning("guard model 加载失败（%s: %s），这一层保持关闭", type(e).__name__, e)
            self._unavailable = True
            return None
        self._model = model
        self._tokenizer = tokenizer
        self._device = next(model.parameters()).device
        self._labels = schema.LABELS
        self._build_judge_prompt = schema.build_judge_prompt
        self._chat_prompt_ids = chat.chat_prompt_ids
        log.info("guard model 加载成功（%s，device=%s）", self.adapter_dir, self._device)
        return self._model

    def _parse_label(self, text: str) -> str:
        """跟 guard/evaluate.py 的 _parse_label 是同一个逻辑：生成的文本里找哪个标签词
        出现了就算哪个，兼容模型多打标点/空格的情况。"""
        low = text.strip().lower()
        for lb in self._labels or ():
            if lb in low:
                return lb
        return "unknown"

    def classify(self, npc_id: str, reply: str) -> str | None:
        """同步、阻塞调用，返回 ok/fabricated/out_of_character/unsafe 之一；
        模型不可用或者推理出了任何异常，返回 None（调用方应当当成"没有额外信息"处理，
        不要当成"ok"——"不可用"和"模型说没问题"是两码事）。

        调用方要负责丢到线程里跑（asyncio.to_thread），这里不做这件事，因为这层
        本身应该是同步、跟 asyncio 无关的，方便脱离服务单独测试。"""
        model = self._load()
        if model is None:
            return None
        import torch

        try:
            prompt = self._build_judge_prompt(_persona_for(npc_id), _full_world_context(), reply)
            ids = self._chat_prompt_ids(self._tokenizer, [{"role": "user", "content": prompt}])
            input_ids = torch.tensor([ids], device=self._device)
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=6,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                )
            gen_ids = out[0][input_ids.shape[-1] :]
            text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        except Exception as e:  # 推理阶段的问题不该打断游戏，记下来就好
            log.warning("[%s] guard model 推理失败（%s: %s），当作没有这一层的判断", npc_id, type(e).__name__, e)
            return None
        label = self._parse_label(text)
        return label if label in (self._labels or ()) else None
