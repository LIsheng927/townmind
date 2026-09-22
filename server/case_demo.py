"""案件模式（v2，"罗生门"式证词调查）的纯文字验证：不碰 Unity，在终端里跟 5 个 NPC
问话，试一遍能不能在几个人不完整、有偏差的说法之间找矛盾、找印证，拼出真相。这是
"先在后端把核心玩法验证一遍，再动 Unity"这个思路的落地——跟当初验证玩家输入闭环时
先用 web_demo 而不是直接上 Unity 是同一个道理。

用之前确保 server/.env 配好了真实 LLM（TOWNMIND_LLM_PROVIDER + 对应的 key）——案件模式
的记忆转变判断全靠大模型现场判断，没有像小镇那样的规则兜底可用，没有真实模型就没法测。
另外配了 OPENAI_API_KEY 的话，记忆检索会真的走语义相关度，没配就退化成"新近度+重要度"
这套（不影响能不能玩，只是少了语义这一项）。

用法（在 server 目录下）：
  uv run python case_demo.py
交互：
  「名字: 要说的话」            跟对应 NPC 说一句话，比如「老周: 账本上的钱好像不太对吧？」
  「状态」                      看现在谁的哪条记忆已经转变、谁还有哪些待触发
  「证据板」                    按案情节点分组，看你已经从各人那听到的说法——同一个节点下
                                如果有好几条内容还不一样，就是要你自己去比对的地方
  「结案」                      不等问话轮数用完，现在就结案看结局
  「退出」                      直接离开，不看结局
"""
import asyncio

from townmind.case import MISSING_PERSON_CASE, CaseSession
from townmind.llm.embeddings import make_embedder
from townmind.llm.factory import make_client


async def main() -> None:
    llm = make_client()
    if llm is None:
        print("没有配置 LLM（检查 server/.env）——案件模式必须有真实模型才能跑，NPC 的记忆"
              "转变判断全靠大模型，没有规则兜底可用。")
        return
    embedder = make_embedder()  # 没配 OPENAI_API_KEY 时是 None，记忆检索自动退化，不影响能不能玩

    case = MISSING_PERSON_CASE
    session = CaseSession(case, llm, embedder=embedder)

    print(f"=== {case.title} ===")
    print(case.room)
    print(case.background)
    print("在场的人：" + "、".join(f"{n.name}（{n.role}）" for n in case.npcs))
    print(
        f"最多可以问 {session.max_turns} 轮，输入「老周: 你要说的话」这样跟人问话，"
        "「状态」看进度，「证据板」看已收集的证词，「结案」提前结束，「退出」离开。\n"
    )

    while not session.ended:
        try:
            raw = input("> ").strip()
        except EOFError:
            break
        if not raw:
            continue
        if raw in ("退出", "exit", "quit"):
            return
        if raw in ("状态", "status"):
            for n in case.npcs:
                st = session.npc_status(n.id)
                print(f"  {st['name']}：已转变 {st['transitioned'] or '（无）'}，待触发 {st['pending'] or '（无）'}")
            print(f"  剩余问话轮数：{session.time_left()}")
            continue
        if raw in ("证据板", "evidence", "线索"):
            board = session.evidence_board()
            if not board:
                print("  还没收集到任何证词。")
            for fact_id, entries in board.items():
                print(f"  [{fact_id}]")
                for e in entries:
                    mark = "（已纠正/坦白）" if e["state"] == "transitioned" else f"（{e['accuracy']}）"
                    print(f"    {e['npc']} {mark}：{e['text']}")
            continue
        if raw in ("结案", "结束", "done"):
            break
        sep = "：" if "：" in raw else ":"
        if sep not in raw:
            print("格式不对，试试「老周: 账本的事你说清楚」这样")
            continue
        name, _, text = raw.partition(sep)
        target = next((n for n in case.npcs if n.name == name.strip()), None)
        if target is None:
            print("在场的人里没有叫这个名字的，在场：" + "、".join(n.name for n in case.npcs))
            continue
        reply = await session.talk(target.id, text.strip())
        print(f"{target.name}：{reply}")

    ending = session.resolve()
    print(f"\n=== 结局：{ending.title} ===")
    print(ending.text)
    print(f"\n发现的事实：{sorted(session.discovered) or '（无）'}")
    print(f"统计：{dict(session.stats)}")


if __name__ == "__main__":
    asyncio.run(main())
