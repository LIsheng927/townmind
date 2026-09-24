"""让真实模型给每个 NPC 写一份日程，打印出来看符不符合人设。

evals/run.py 量的是行为指标（去过几个地点、在该在的地方占比），回答不了"日程本身像不像
这个人"——那要靠人看。这个脚本就是把 10 份日程拿出来给人看的，顺便记录归一化（coerce_places）
救回了几段、退回模板几份。10 次调用，几秒钟。

用法（在 server 目录下）：
  uv run python -m evals.plan_samples
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path

from townmind.llm.factory import make_client
from townmind.personas import PERSONAS
from townmind.planner import WRITE_PLAN_TOOL, DailyPlan, coerce_places, plan_prompt

RESULTS_DIR = Path(__file__).parent / "results"


async def main() -> None:
    llm = make_client()
    if llm is None:
        raise SystemExit("没有可用的大模型：请检查 server/.env。")
    rows = []
    for npc_id, p in PERSONAS.items():
        if npc_id == "player":
            continue
        system, user = plan_prompt(npc_id)
        call = await llm.choose_tool(system, user, [WRITE_PLAN_TOOL])
        raw_places = [b.get("place") for b in call.arguments.get("blocks", [])]
        coerced = coerce_places(call.arguments, npc_id)
        fixed = sum(1 for a, b in zip(raw_places, [x["place"] for x in coerced["blocks"]]) if a != b)
        dropped = len(raw_places) - len(coerced["blocks"])
        try:
            plan = DailyPlan(**coerced)
            ok, text = True, plan.describe()
        except Exception as e:
            ok, text = False, f"校验失败：{type(e).__name__}: {str(e)[:80]}"
        rows.append({"npc": npc_id, "name": p["name"], "home": p.get("home", ""), "ok": ok,
                     "fixed_places": fixed, "dropped_blocks": dropped, "plan": text, "raw": call.arguments})
        print(f"[{p['name']:6}] 工作地点={p.get('home') or '无':4} 归一化修了 {fixed} 段、丢了 {dropped} 段 {'' if ok else '(退回模板)'}\n        {text}")
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}-plan-samples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md = "\n".join(f"- **{r['name']}**（{r['home'] or '无固定地点'}）：{r['plan']}" for r in rows)
    (RESULTS_DIR / f"{stamp}-plan-samples.md").write_text(md + "\n", encoding="utf-8")
    print(f"\n已保存到 evals/results/{stamp}-plan-samples.(json|md)")


if __name__ == "__main__":
    asyncio.run(main())
