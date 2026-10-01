"""记忆的异步审核：用一个更强的模型，离线、批量地重审"NPC 自己说过的话"。

要解决的问题（复习项目时自己发现的一个结构性漏洞）：

  输出检查（前门）和回忆核对（后门）用的是同一个本地 guard 分类器。一句编造的话
  如果前门没拦住、被当成"你对X说了「…」"存进了记忆，那后门再拿同一个模型看一遍，
  绝大多数情况下还是"ok"——同一个模型不会因为多问一次就变聪明。于是后门实际上只对
  "guard 关着/用的是旧版本时存进来的记忆"有用；guard 开着时漏进来的编造，会在记忆里
  一直以"ok"的身份被反复想起、反复当成事实说出去。大样本回测里 guard-v2 的抓住率是
  85%~95%，也就是说每存一百条自己说的话，有五到十五条编造会以这种方式长期潜伏。

这里的做法：

  * 决策链路上不动——guard 还是那个 guard，快、便宜、本地。
  * 另开一条慢路：后台每隔一段时间，把还没审过的"自己说过的话"攒成一批，连同设定、
    人设、这个 NPC 听别人说过的话（作为依据），交给一个强模型（默认 gpt-5.5 /
    claude-sonnet-4-5，怎么定的见 DEFAULT_AUDIT_MODELS 上的注释）逐条判 ok / suspect，结论写回 Memory.audited 并落盘。
  * 决策时按 audited 分三路：suspect → 直接进 suspect_said 注入（"这条你多半记错了"），
    不再问本地 guard；ok → 不问 guard（强模型已经放行，省一次推理）；"" → 维持原样，
    问本地 guard，并在提示词里标一句"自己说的，尚未核实"。

为什么不直接在前门换强模型：前门在决策链路上，每句话都要过，8 秒超时、并发 4，
强模型慢十倍、贵二十倍，放前门直接把延迟和成本顶上去。审核是写后（write-behind）：
存进去的时候先用快模型把关，慢模型在后台慢慢补审，审核结果对下一次回忆生效。
代价是"存进来到被审完"之间有一个窗口期（审核周期，默认 30 秒），窗口期内这条记忆
被想起时只有"尚未核实"的标记和本地 guard 两道弱保护。

成本：审核调用按 NPC 分批、每批最多 AUDIT_BATCH 条，一次调用审一批；一条记忆一生
只审一次。独立计 token（audit_tokens_in/out），别跟主决策混在一起——两边单价差一个
数量级，混在一起算成本就没法看了。

命令行（不启动服务，直接对落盘的记忆文件操作）：

    uv run python -m townmind.auditor data/memories            # 只统计：审过多少、几条 suspect
    uv run python -m townmind.auditor data/memories --run      # 真审：把没审过的都审一遍并写回
    uv run python -m townmind.auditor data/memories --run --reaudit   # 全部重审（换了审核模型时用）

"泄漏率"=审过的"自己说过的话"里被判 suspect 的比例：这就是前门放进去的编造占比，
是本地 guard 在真实运行里（而不是在评测集上）的漏检率的直接估计。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .guard_model import _full_world_context, _persona_for
from .llm.base import LLMClient, ToolCall
from .memory import Memory, MemoryStore

log = logging.getLogger("townmind.auditor")

ENV_ENABLE = "TOWNMIND_MEMORY_AUDIT"
ENV_MODEL = "TOWNMIND_AUDIT_MODEL"
ENV_INTERVAL = "TOWNMIND_AUDIT_INTERVAL"
# openai 这边定 gpt-5.5 是真实数据定的：同一批 90 条、三个模型各审一遍、逐条人工复核，
# gpt-4o 抓 2 错 4、gpt-5.4-mini 抓 2 错 2、gpt-5.5 抓 3 错 1（见 README"写后审核"小节）。
# 审核不在决策链路上，90 条两毛钱，贵这一点换的是少漏少错。anthropic 那边没有跑过对比，沿用裁判的默认
DEFAULT_AUDIT_MODELS = {"openai": "gpt-5.5", "anthropic": "claude-sonnet-4-5"}
AUDIT_BATCH = 20  # 一次调用最多审几条；再多模型容易数错行、输出对不上号
AUDIT_INTERVAL = 30.0  # 后台循环的间隔（秒）
AUDIT_TIMEOUT = 40.0  # 一批 20 条、强模型、要读完整设定，比主决策的 8 秒宽得多
EVIDENCE_LIMIT = 40  # 给审核模型看的"这个 NPC 听别人说过的话"最多几条（最新的）

AUDITED_OK = "ok"
AUDITED_SUSPECT = "suspect"

# 跟 agent._SELF_SAID_RE 同一个正则。不从 agent 导入：agent 是热路径、依赖多，审核器
# 要能在命令行里脱离 Agent 单独跑。
_SELF_SAID_RE = re.compile(r"^你(?:对.*?)?说了「(.*)」$")


def self_said_text(memory_text: str) -> str | None:
    m = _SELF_SAID_RE.match(memory_text)
    return m.group(1) if m else None


class AuditVerdicts(BaseModel):
    """逐条给出结论，按输入顺序一一对应。"""

    verdicts: list[Literal["ok", "suspect"]] = Field(description="每条话的结论，按输入编号顺序一一对应")


AUDIT_TOOL = {
    "name": "audit_memories",
    "description": "逐条判断 NPC 自己说过的话有没有编造设定里不存在的事实",
    "parameters": AuditVerdicts.model_json_schema(),
}

SYSTEM_TEMPLATE = (
    "你是一个游戏小镇 NPC 对话系统的事后审核员。下面是小镇的完整设定和这个 NPC 的人设，"
    "再下面是这个 NPC 以前说过的一些话。请逐条判断：这句话有没有**以肯定的口吻**把设定里不存在的"
    "具体事实（人物、地点、物品、数量、价格、事件、规矩、历史、别人的隐私、未来的安排）当成真事说出来。\n\n"
    "口径（按顺序套，命中一条就定）：\n"
    "1. 问句一律 ok——问别人、反问、打听，都不是在陈述事实。\n"
    "2. 转述一律 ok——带「听说」「据说」「有人说」「某某说」的，或者内容在下面「听别人说过的话」里出现过的，"
    "哪怕跟设定冲突也判 ok：这是在传话，不是在编；传话的问题由别的机制处理。\n"
    "3. 纠正一律 ok——说「没有这回事」「镇上可没有某某」而设定确实没有的，是在纠正别人，不是编造。\n"
    "4. 设定原文或人设里写着的、能从中直接推出来的，ok。\n"
    "5. 寒暄、问候、情绪、自己的工作和心情这类人设范围内的日常，ok。\n"
    "6. 剩下的：以肯定口吻说了一件设定和依据里都没有的具体事情（比如一个具体的数目、价格、规定、"
    "某人的私事、某件过去或将来的事）——suspect。\n"
    "拿不准的归 ok——这个审核的目的是找出明显编造的那些，不是把一切没写进设定的话都当成编造；"
    "日常闲聊里大量内容本来就不在设定里。\n\n"
    "【小镇设定】\n{world}\n\n【这个 NPC】\n{persona}\n\n"
    "【这个 NPC 听别人说过的话，可以作为依据】\n{evidence}"
)


def make_audit_client() -> LLMClient | None:
    """跟 llm.factory.make_client 同一套 provider/key，只是换成审核模型（默认比 NPC 强一档）。
    用同一个模型审自己说的话，偏差会往同一个方向倒——跟评测裁判的道理一样。"""
    from dotenv import load_dotenv

    from .llm.factory import ENV_PATH

    load_dotenv(ENV_PATH)
    provider = os.getenv("TOWNMIND_LLM_PROVIDER", "anthropic").lower()
    model = os.getenv(ENV_MODEL) or DEFAULT_AUDIT_MODELS.get(provider)
    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        from .llm.anthropic_client import AnthropicClient

        return AnthropicClient(model)
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        from .llm.openai_client import OpenAIClient

        client = OpenAIClient(model)
        # 审核输出本身很短（20 个 ok/suspect），但换成 GPT-5 这类推理模型时，推理 token 也算在
        # 输出上限里，NPC 台词用的 300 会把推理截断、最后一个 tool call 都吐不出来
        client.max_tokens = 4000
        return client
    return None


def pending_in(store: MemoryStore, reaudit: bool = False) -> list[Memory]:
    """这个 store 里还没审过的"自己说过的话"，按时间先后。reaudit=True 时全部重审。"""
    return sorted(
        (m for m in store.memories if self_said_text(m.text) is not None and (reaudit or not m.audited)),
        key=lambda m: m.time,
    )


def evidence_lines(store: MemoryStore, limit: int = EVIDENCE_LIMIT) -> list[str]:
    """给审核模型当依据的记忆：这个 NPC 听别人说的、见到的，不包括它自己说的（自己说的
    正是待审对象，拿它当依据就循环论证了）、也不包括反思（反思是从自己的记忆里提炼的，
    编造会顺着反思被"洗白"）。"""
    others = [
        m for m in store.memories
        if self_said_text(m.text) is None and m.kind == "event"
    ]
    others.sort(key=lambda m: m.time, reverse=True)
    return [m.text for m in others[:limit]]


class MemoryAuditor:
    """把 agent 里每个 NPC 还没审过的"自己说过的话"交给强模型重审，结论写回并落盘。

    agent 只要求鸭子类型地提供 _memories（npc_id -> MemoryStore）、_memory_path(npc_id)、
    stats；命令行模式下没有 Agent，直接传 store 字典进 audit_stores()。"""

    def __init__(self, llm: LLMClient, batch_size: int = AUDIT_BATCH, timeout: float = AUDIT_TIMEOUT) -> None:
        self.llm = llm
        self.batch_size = batch_size
        self.timeout = timeout
        self.stats: dict[str, int] = defaultdict(int)

    # ---- 核心：审一批 ----
    async def audit_batch(self, npc_id: str, store: MemoryStore, batch: list[Memory]) -> bool:
        """审这一批，把结论写进 memory.audited。返回这批有没有审成；审不成就原样留着，
        下一轮再来——审核是尽力而为的后台工作，失败不能影响任何在线行为。"""
        if not batch:
            return True
        texts = [self_said_text(m.text) or m.text for m in batch]
        system = SYSTEM_TEMPLATE.format(
            world=_full_world_context(),
            persona=_persona_for(npc_id),
            evidence="\n".join(f"- {t}" for t in evidence_lines(store)) or "（暂无）",
        )
        user = "请逐条判断下面这些话（都是这个 NPC 自己说过的）：\n" + "\n".join(
            f"{i + 1}. {t}" for i, t in enumerate(texts)
        )
        self.stats["audit_calls"] += 1
        try:
            call: ToolCall = await asyncio.wait_for(self.llm.choose_tool(system, user, [AUDIT_TOOL]), self.timeout)
        except Exception as e:
            self.stats["audit_failures"] += 1
            log.warning("[%s] 记忆审核调用异常（%s: %s），这批下次再审", npc_id, type(e).__name__, e)
            return False
        self.stats["audit_tokens_in"] += call.input_tokens
        self.stats["audit_tokens_out"] += call.output_tokens
        try:
            verdicts = AuditVerdicts(**call.arguments).verdicts
        except Exception as e:
            self.stats["audit_failures"] += 1
            log.warning("[%s] 记忆审核结果格式不对（%s: %s），这批下次再审", npc_id, type(e).__name__, e)
            return False
        if len(verdicts) != len(batch):
            # 数量对不上就整批作废，不猜"前几条对得上"——错位一条，后面全错，而且是静悄悄地错
            self.stats["audit_failures"] += 1
            log.warning("[%s] 记忆审核结果数量（%d）跟送审数量（%d）对不上，这批下次再审", npc_id, len(verdicts), len(batch))
            return False
        for m, v in zip(batch, verdicts):
            m.audited = AUDITED_SUSPECT if v == "suspect" else AUDITED_OK
            self.stats["audited_suspect" if v == "suspect" else "audited_ok"] += 1
            if v == "suspect":
                log.info("[%s] 审核判定为编造：%s", npc_id, self_said_text(m.text))
        return True

    async def audit_stores(
        self, stores: dict[str, MemoryStore], paths: dict[str, Path | None] | None = None,
        max_batches: int | None = None, reaudit: bool = False,
    ) -> int:
        """把这些 store 里待审的都审一遍；每个 store 审完就落盘。返回这次审了多少条。
        max_batches 限制一次最多发几批调用——后台循环用它防止某一轮积压太多把这一轮拖得很长。"""
        done, batches = 0, 0
        for npc_id, store in list(stores.items()):
            todo = pending_in(store, reaudit=reaudit)
            if not todo:
                continue
            changed = False
            for i in range(0, len(todo), self.batch_size):
                if max_batches is not None and batches >= max_batches:
                    break
                batch = todo[i:i + self.batch_size]
                batches += 1
                if await self.audit_batch(npc_id, store, batch):
                    done += len(batch)
                    changed = True
            path = (paths or {}).get(npc_id)
            if changed and path is not None:
                try:
                    store.save(path)
                except OSError as e:
                    log.warning("[%s] 审核结果存盘失败：%s", npc_id, e)
        return done

    async def run_once(self, agent: Any, max_batches: int | None = 3) -> int:
        """给服务用：审 agent 里已经加载的所有 NPC。统计并进 agent.stats，/stats 里能看到。"""
        stores = dict(agent._memories)
        paths = {n: agent._memory_path(n) for n in stores}
        before = dict(self.stats)
        done = await self.audit_stores(stores, paths, max_batches=max_batches)
        for k, v in self.stats.items():
            agent.stats[k] += v - before.get(k, 0)
        return done

    async def run_forever(self, agent: Any, interval: float = AUDIT_INTERVAL) -> None:
        """后台循环：每隔 interval 秒审一轮。任何异常只记日志不退出——这条循环死了，
        审核就静悄悄停了，比抛出来更糟。"""
        while True:
            try:
                await asyncio.sleep(interval)
                if not getattr(agent, "memory_audit", True):
                    continue  # 运行时把开关关了：循环还活着，只是不干活，开回来立刻恢复
                n = await self.run_once(agent)
                if n:
                    log.info("记忆审核：这轮审了 %d 条（累计 suspect %d / ok %d）", n, self.stats["audited_suspect"], self.stats["audited_ok"])
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("记忆审核循环异常（%s: %s），下一轮继续", type(e).__name__, e)


# ---- 命令行：对落盘的记忆文件做统计 / 离线审核 ----
def leak_report(stores: dict[str, MemoryStore]) -> dict:
    """每个 NPC：自己说过的话多少条、审过多少、几条 suspect；汇总一个泄漏率。"""
    per = {}
    total_said = total_audited = total_suspect = 0
    for npc_id, store in sorted(stores.items()):
        said = [m for m in store.memories if self_said_text(m.text) is not None]
        audited = [m for m in said if m.audited]
        suspect = [m for m in audited if m.audited == AUDITED_SUSPECT]
        per[npc_id] = {
            "said": len(said), "audited": len(audited), "suspect": len(suspect),
            "suspect_texts": [self_said_text(m.text) for m in suspect],
        }
        total_said += len(said)
        total_audited += len(audited)
        total_suspect += len(suspect)
    return {
        "per_npc": per,
        "said": total_said,
        "audited": total_audited,
        "suspect": total_suspect,
        "leak_rate": (total_suspect / total_audited) if total_audited else None,
    }


def _load_dir(memory_dir: Path) -> tuple[dict[str, MemoryStore], dict[str, Path]]:
    stores, paths = {}, {}
    for path in sorted(memory_dir.glob("*.json")):
        if path.name.endswith(".social.json"):
            continue  # 社交关系文件（social.py）跟记忆放在同一个目录，不是记忆
        npc_id = path.stem
        stores[npc_id] = MemoryStore.load(path)
        paths[npc_id] = path
    return stores, paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="离线审核 NPC 记忆里'自己说过的话'，或只统计泄漏率")
    ap.add_argument("memory_dir", type=Path, help="记忆目录，通常是 server/data/memories")
    ap.add_argument("--run", action="store_true", help="真的调用审核模型；不加只统计")
    ap.add_argument("--reaudit", action="store_true", help="已审过的也重审（换了审核模型时用）")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出报告")
    ap.add_argument("--nli", action="store_true", help="对每条 suspect 再算一遍 NLI 证据否决会不会推翻它（只报告，不改文件）")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    stores, paths = _load_dir(args.memory_dir)
    if not stores:
        print(f"{args.memory_dir} 里没有记忆文件", file=sys.stderr)
        return 1
    if args.run:
        llm = make_audit_client()
        if llm is None:
            print("没配审核模型（需要 .env 里的 API key；模型用 TOWNMIND_AUDIT_MODEL 指定）", file=sys.stderr)
            return 2
        auditor = MemoryAuditor(llm)
        n = asyncio.run(auditor.audit_stores(stores, paths, reaudit=args.reaudit))
        print(f"审了 {n} 条；调用 {auditor.stats['audit_calls']} 次，失败 {auditor.stats['audit_failures']} 次，"
              f"tokens in/out {auditor.stats['audit_tokens_in']}/{auditor.stats['audit_tokens_out']}")
    report = leak_report(stores)
    nli_notes: dict[str, str] = {}
    if args.nli:
        # 服务里 suspect 进注入之前要过的那道保险，这里离线算一遍给人看：哪条会被设定推翻。
        # 不写回 audited——否决是回忆时现算的，不是审核结论的一部分
        from .grounding import VETO_THRESHOLD, GroundingChecker

        g = GroundingChecker()
        if not g.available:
            print("NLI 不可用：先 `uv sync --group guard-model`", file=sys.stderr)
            return 2
        for npc_id, r in report["per_npc"].items():
            for t in r["suspect_texts"]:
                score = g.supported(npc_id, t)
                verdict = "-" if score is None else ("推翻" if score >= VETO_THRESHOLD else "维持")
                nli_notes[t] = f"  NLI 蕴含 {score:.2f} → {verdict}" if score is not None else "  NLI -"
        report["nli_overruled"] = sum(1 for n in nli_notes.values() if n.endswith("推翻"))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    print(f"{'NPC':<12}{'自己说的':>8}{'已审':>6}{'suspect':>9}")
    for npc_id, r in report["per_npc"].items():
        print(f"{npc_id:<12}{r['said']:>8}{r['audited']:>6}{r['suspect']:>9}")
        for t in r["suspect_texts"]:
            print(f"    ✗ {t}{nli_notes.get(t, '')}")
    rate = report["leak_rate"]
    print(f"合计：自己说的 {report['said']} 条，已审 {report['audited']} 条，suspect {report['suspect']} 条，"
          f"泄漏率 {rate:.1%}" if rate is not None else
          f"合计：自己说的 {report['said']} 条，还没有审过任何一条（加 --run 开始审）")
    if args.nli:
        print(f"NLI 证据否决会推翻其中 {report['nli_overruled']} 条（回忆时现算，不改审核结论）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
