"""记忆审核器（townmind/auditor.py）自己的评测：召回率和误伤率。

为什么要有这个：真实运行的 90 条"自己说过的话"上，三个审核模型各抓 2/2/3 条编造、错 4/2/1 条
（README"写后审核"小节）——但那只能算"精确率"，算不出召回：83 条判 ok 的里面有没有漏掉的编造，
没人逐句核过，核了也只是一个人的口径。要量召回，得有**事先知道答案**的题。

做法：拿 guard 分类器那两份人工过过的场景库（evals/scenarios/guard.json 111 条、guard-holdout.json
94 条，每条带 npc_id / reply / expected），把每条 reply 当成"这个 NPC 自己说过的话"种进它的记忆
（默认种进 data/memories 里真实运行攒下的记忆，这样审核模型看到的依据和真实服务里一样；--empty
种进空记忆），然后让审核器审。expected=fabricated 的被判 suspect 算抓住，expected=ok 的被判
suspect 算误伤。按 kind 分组看哪类编造它抓得住、哪类抓不住。

跟 guard 分类器的对照：同一份场景库，guard-v2 的数字在 evals/guard_fabrication.py 的结果里
（111 条：抓住 92% / 误伤 2%；94 条主题事先定死的：抓住 74% / 误伤 2%）。审核器是后门、慢路，
理应比前门的本地小模型强——强多少，这个脚本量。两边输入格式不同（审核器多看人设、依据，
而且是 20 条一批），不是严格的同台比较，但同一批题、同一个口径。

真实记忆里原有的"自己说过的话"不会被送审（它们已经审过、有 audited 字段；--reaudit-existing
才会一起重审），所以每次只花种进去的这几十条的钱——94 条、5 批、不到两毛。什么都不落盘。

用法（在 server 目录下）：
  uv run python -m evals.audit_recall                                   # guard-holdout.json 94 条，默认审核模型
  uv run python -m evals.audit_recall --scenarios evals/scenarios/guard.json
  uv run python -m evals.audit_recall --model gpt-5.4-mini             # 换审核模型
  uv run python -m evals.audit_recall --empty                           # 不用真实记忆当依据
  uv run python -m evals.audit_recall --model gpt-5.4-mini --nli       # 再加一道 NLI 证据否决（服务里就是这么组合的）

--nli 做的事跟 agent._flag_suspect_said_memories 里一样：审核器判 suspect 的，再用 NLI 看设定/人设能不能推出
这句话，蕴含过 VETO_THRESHOLD 就推翻。要装 guard-model 依赖组（第一次会下载 2.2GB 的 XNLI 模型）。
结果表多一行"审核器 + NLI 否决"：看它救回几条误伤、又误放几条编造。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from townmind.auditor import ENV_MODEL, AUDITED_SUSPECT, MemoryAuditor, make_audit_client
from townmind.grounding import VETO_THRESHOLD, GroundingChecker
from townmind.memory import Memory, MemoryStore

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_BANK = Path(__file__).parent / "scenarios" / "guard-holdout.json"
DEFAULT_MEMORY_DIR = Path(__file__).resolve().parents[1] / "data" / "memories"


def load_bank(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("reviewed"):
        print(f"警告：{path.name} 标着 reviewed=false，没人看过的题只能跑流程，不能拿来下结论。")
    return data["items"]


def load_stores(memory_dir: Path | None) -> dict[str, MemoryStore]:
    stores: dict[str, MemoryStore] = {}
    if memory_dir is not None:
        for p in sorted(memory_dir.glob("*.json")):
            if p.name.endswith(".social.json"):
                continue
            stores[p.stem] = MemoryStore.load(p)
    return stores


def plant(stores: dict[str, MemoryStore], items: list[dict], now: float, reaudit_existing: bool = False) -> list[tuple[dict, Memory]]:
    """把每条题当成"你对玩家说了「reply」"种进对应 NPC 的记忆（只在内存里）。返回 (题, 记忆) 对。"""
    if not reaudit_existing:
        # 原有没审过的先按 ok 封存，免得这次把钱花在它们身上。必须在种题之前做完，
        # 不然种进去的题会被后一轮封存一起盖掉（踩过：一个 NPC 种第二条时把第一条封成了 ok）
        for store in stores.values():
            for m in store.memories:
                if not m.audited:
                    m.audited = "ok"
    planted = []
    for it in items:
        store = stores.setdefault(it["npc_id"], MemoryStore())
        store.add(f"你对玩家说了「{it['reply']}」", 4, now, people={"player"})
        planted.append((it, store.memories[-1]))
    return planted


def apply_nli(rows: list[dict], grounding: GroundingChecker) -> None:
    """给每行补上 NLI 否决之后的结论：flagged 且设定能推出 → 推翻。只对 flagged 的算（跟服务里一样，
    NLI 只在判编造时出面），没 flagged 的 nli_score 留 None。"""
    for r in rows:
        r["nli_score"] = None
        r["flagged_after_nli"] = r["flagged"]
        if r["flagged"]:
            score = grounding.supported(r["npc_id"], r["reply"])
            r["nli_score"] = score
            if score is not None and score >= VETO_THRESHOLD:
                r["flagged_after_nli"] = False


def summarize(planted: list[tuple[dict, Memory]], auditor: MemoryAuditor, model: str, bank: Path, base: str,
              grounding: GroundingChecker | None = None) -> tuple[dict, list[dict]]:
    rows = []
    for it, m in planted:
        rows.append({
            "npc_id": it["npc_id"], "reply": it["reply"], "expected": it["expected"],
            "kind": it.get("kind", ""), "audited": m.audited,
            "flagged": m.audited == AUDITED_SUSPECT,
        })
    if grounding is not None:
        apply_nli(rows, grounding)
    fab = [r for r in rows if r["expected"] == "fabricated"]
    ok = [r for r in rows if r["expected"] == "ok"]
    unaudited = [r for r in rows if not r["audited"]]
    by_kind = {}
    for kind, group in _group_by_kind(rows).items():
        f = [r for r in group if r["expected"] == "fabricated"]
        o = [r for r in group if r["expected"] == "ok"]
        by_kind[kind] = {
            "n": len(group),
            "recall": (sum(r["flagged"] for r in f) / len(f)) if f else None,
            "false_positive": (sum(r["flagged"] for r in o) / len(o)) if o else None,
        }
    summary = {
        "bank": bank.name, "model": model, "base": base, "n": len(rows),
        "fabricated": len(fab), "ok": len(ok),
        "recall": (sum(r["flagged"] for r in fab) / len(fab)) if fab else None,
        "false_positive": (sum(r["flagged"] for r in ok) / len(ok)) if ok else None,
        "nli": grounding is not None,
        "recall_nli": (sum(r["flagged_after_nli"] for r in fab) / len(fab)) if (fab and grounding is not None) else None,
        "false_positive_nli": (sum(r["flagged_after_nli"] for r in ok) / len(ok)) if (ok and grounding is not None) else None,
        "nli_rescued": sum(1 for r in rows if grounding is not None and r["flagged"] and not r["flagged_after_nli"] and r["expected"] == "ok"),
        "nli_let_through": sum(1 for r in rows if grounding is not None and r["flagged"] and not r["flagged_after_nli"] and r["expected"] == "fabricated"),
        "unaudited": len(unaudited),
        "by_kind": by_kind,
        "audit_calls": auditor.stats["audit_calls"], "audit_failures": auditor.stats["audit_failures"],
        "tokens_in": auditor.stats["audit_tokens_in"], "tokens_out": auditor.stats["audit_tokens_out"],
    }
    return summary, rows


def _group_by_kind(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("kind"):
            groups.setdefault(r["kind"], []).append(r)
    return groups


def _pct(v) -> str:
    return "-" if v is None else f"{v:.0%}"


def render_table(s: dict) -> str:
    lines = [
        f"审核模型 `{s['model']}`，场景库 {s['bank']}（{s['n']} 条：编造 {s['fabricated']}、真话/寒暄 {s['ok']}），"
        f"依据 = {s['base']}；{s['audit_calls']} 次调用、失败 {s['audit_failures']}，"
        f"token in/out {s['tokens_in']}/{s['tokens_out']}"
        + (f"；**{s['unaudited']} 条没审成**（调用失败那几批）" if s["unaudited"] else ""),
        "",
        "| | 编造抓住率（召回） | 真话误伤率 |",
        "|---|---|---|",
        f"| 审核器（{s['model']}） | **{_pct(s['recall'])}** | **{_pct(s['false_positive'])}** |",
    ]
    if s.get("nli"):
        lines.append(
            f"| 审核器 + NLI 证据否决 | **{_pct(s['recall_nli'])}** | **{_pct(s['false_positive_nli'])}** |"
        )
        lines.append("")
        lines.append(f"NLI 救回误伤 {s['nli_rescued']} 条，误放编造 {s['nli_let_through']} 条（阈值 {VETO_THRESHOLD}）。")
    lines += [
        "",
        "| 题型 | 条数 | 编造抓住率 | 真话误伤率 |",
        "|---|---|---|---|",
    ]
    for kind, k in s["by_kind"].items():
        lines.append(f"| {kind.split('（')[0]} | {k['n']} | {_pct(k['recall'])} | {_pct(k['false_positive'])} |")
    return "\n".join(lines)


def render_rows(rows: list[dict]) -> str:
    out = []
    for r in rows:
        mark = "✓" if r["flagged"] == (r["expected"] == "fabricated") else "✗"
        nli = ""
        if r.get("nli_score") is not None:
            verdict = "推翻" if not r["flagged_after_nli"] else "维持"
            nli = f"  NLI 蕴含 {r['nli_score']:.2f} → {verdict}"
        out.append(f"{mark} [{r['expected']:<10}] {r['npc_id']:<6} {r['audited'] or '(未审)':<8} {r['reply']}{nli}")
    return "\n".join(out)


async def run(items: list[dict], memory_dir: Path | None, model: str | None, reaudit_existing: bool,
              nli: bool = False) -> tuple[dict, list[dict]]:
    if model:
        os.environ[ENV_MODEL] = model
    llm = make_audit_client()
    if llm is None:
        print("没配审核模型（需要 .env 里的 API key）", file=sys.stderr)
        sys.exit(2)
    grounding = None
    if nli:
        grounding = GroundingChecker()
        if not grounding.available:
            print("NLI 不可用：先 `uv sync --group guard-model`（第一次会下载约 2.2GB 的 XNLI 模型）", file=sys.stderr)
            sys.exit(2)
    auditor = MemoryAuditor(llm)
    stores = load_stores(memory_dir)
    planted = plant(stores, items, time.time(), reaudit_existing)
    await auditor.audit_stores(stores, paths={}, reaudit=False)  # paths 为空：什么都不落盘
    return summarize(planted, auditor, getattr(llm, "model", str(model)), Path("bank"),
                     "空记忆" if memory_dir is None else f"真实记忆 {memory_dir.name}/", grounding)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", type=Path, default=DEFAULT_BANK, help="场景库 JSON（默认 guard-holdout.json 94 条）")
    ap.add_argument("--model", default=None, help=f"审核模型；不传用 {ENV_MODEL} 或默认")
    ap.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR, help="种进哪份真实记忆当依据")
    ap.add_argument("--empty", action="store_true", help="种进空记忆，不用真实记忆当依据")
    ap.add_argument("--reaudit-existing", action="store_true", help="真实记忆里原有的自己说过的话也一起重审（多花钱）")
    ap.add_argument("--nli", action="store_true", help="审核器判 suspect 的再过一道 NLI 证据否决（跟服务里的组合一样）")
    args = ap.parse_args()
    items = load_bank(args.scenarios)
    summary, rows = asyncio.run(run(items, None if args.empty else args.memory_dir, args.model, args.reaudit_existing, args.nli))
    summary["bank"] = args.scenarios.name
    table = render_table(summary)
    print(f"\n{table}\n\n{render_rows(rows)}")
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}-audit-recall.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (RESULTS_DIR / f"{stamp}-audit-recall.md").write_text(table + "\n\n```\n" + render_rows(rows) + "\n```\n", encoding="utf-8")
    print(f"\n已保存到 evals/results/{stamp}-audit-recall.(json|md)")


if __name__ == "__main__":
    main()
