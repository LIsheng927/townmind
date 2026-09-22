"""八卦传播实验：一条消息在小镇里能传多远、传到后面还剩几分可信。

前置是 townmind/social.py（好感/信任）和 memory.py 里的 shareable()/hop：NPC 会主动把
自己知道的、对方多半还不知道的事讲出去，听到的人记成"第 N 手消息"，重要度按代数打折，
提示词里标成"辗转听来的传闻"。这些机制单独看都有单测覆盖，但有两个问题是看代码看不出来、
必须实际跑一遍才知道答案的：

  1. 一条消息到底能传几手？"重要度按代数打折"和"重要度不够就不值得说"这两条规则凑在一起，
     消息应该会自己"传死"——但具体死在第几手，取决于 HOP_IMPORTANCE_DECAY 和
     SHARE_MIN_IMPORTANCE 这两个数怎么配，拍脑袋算不如直接量。
  2. 信任门槛到底拦住了多少？trust < SHARE_TRUST_FLOOR 的人不会被告知，这在单测里只能
     验证"确实拦住了这一次"，拦不拦得住"整个小镇的扩散"是另一回事。

这个脚本不调用大模型：它模拟的是"谁把什么讲给了谁"这一层的机制，用的就是 MemoryStore
和 RelationshipBook 本身的代码（shareable / mark_told / retold_importance / trust_of），
不另外搭一套简化逻辑——这一点跟 evals/memory_ablation.py 第四档"recall() 实际排序"的取向
一致：要验证的是真实代码路径，不是我对它的转述。大模型那一层（具体措辞、要不要顺口提
一句）不在这个实验的范围内，它影响的是"说得像不像人"，不是"消息能传多远"。

这个脚本第一次跑出来就抓到一个光看代码看不出来的问题：不管初始重要度是 5 还是 10，
任何消息都只能传 1 手。原因是听到的话一律按 IMPORTANCE_HEARD(=6) 拍平记进记忆，再按
衰减打一折变成 5，就掉到 SHARE_MIN_IMPORTANCE(=6) 以下，于是第二个人永远不会再往下说。
"越轰动的消息传得越远"这个本来想要的性质，在真实链路上根本不成立——传播机制基本等于
没有。改法是让听者继承"转述者自己觉得这事有多重要"（以 IMPORTANCE_HEARD 为下限）再折
一手，而不是拍平。改完之后重要度 5 的消息传 0 手、6 传 1 手、7~8 传 2 手、9~10 传 3 手，
才真的拉开了差距。tests/test_agent.py 里
test_important_news_is_still_worth_passing_on_after_one_retelling 是这个问题的回归测试。

用法（在 server 目录下）：
  uv run python -m evals.gossip_propagation
"""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from statistics import mean

from townmind.agent import IMPORTANCE_HEARD, SHARE_TRUST_FLOOR
from townmind.memory import (
    HOP_IMPORTANCE_DECAY,
    SHARE_MIN_IMPORTANCE,
    MemoryStore,
    retold_importance,
)
from townmind.social import RelationshipBook

RESULTS_DIR = Path(__file__).parent / "results"
NOW = 10_000.0
RUMOR = "集市的米价涨了三成"
MAX_HOPS = 8  # 防止万一哪天参数配得让消息永远传不死，这里兜一个底


def tell(teller: MemoryStore, listener: MemoryStore, teller_id: str, listener_id: str, now: float):
    """模拟一次"把知道的事讲给下一个人听"，逐条对应 agent.py 里的真实流程：

      _pick_share_hint  -> teller.shareable(listener_id, now, k=1)
      decide() 说话之后  -> teller.mark_told(m, listener_id)
      _remember 收下这句 -> listener.add(..., retold_importance(...), hop=m.hop + 1)

    讲不出来（没有够分量的、或者都跟对方讲过了）就返回 None。"""
    picked = teller.shareable(listener_id, now, k=1)
    if not picked:
        return None
    m = picked[0]
    teller.mark_told(m, listener_id)
    # 跟 _remember 一致：以 IMPORTANCE_HEARD 为下限继承转述者的判断，再折一手
    importance = retold_importance(max(IMPORTANCE_HEARD, m.importance))
    listener.add(f"{teller_id}对你说：「{RUMOR}」", importance, now, {teller_id}, hop=m.hop + 1)
    return listener.memories[-1]


def chain(origin_importance: int) -> list[dict]:
    """一条消息沿着一串 NPC 往下传，记录每一手的状态，直到传不动为止。"""
    holder = MemoryStore()
    holder.add(RUMOR, origin_importance, NOW)  # 第 0 手：亲历者
    steps = [
        {
            "hop": 0,
            "importance": holder.memories[-1].importance,
            "worth_passing_on": holder.memories[-1].importance >= SHARE_MIN_IMPORTANCE,
            "flagged_as_hearsay": False,
        }
    ]
    for i in range(MAX_HOPS):
        listener = MemoryStore()
        got = tell(holder, listener, f"npc{i}", f"npc{i + 1}", NOW)
        if got is None:
            break
        steps.append(
            {
                "hop": got.hop,
                "importance": got.importance,
                "worth_passing_on": got.importance >= SHARE_MIN_IMPORTANCE,
                "flagged_as_hearsay": got.hop >= 1,  # 提示词里会标"辗转听来的传闻"
            }
        )
        holder = listener
    return steps


def spread_once(n: int, seed: int, trust_gate: bool, rounds: int, degree: int) -> tuple[int, int, int]:
    """一条消息在 n 个人的小镇里扩散一次，返回（最终知道的人数、被不信任挡下的次数、传到第几手）。

    小镇排成一个环，每个人只跟左右各 degree//2 个邻居打照面——不是"人人都能碰到人人"。
    这一点很关键：第一版就是写成全连通的，结果不管看不看信任，最后都是 100% 的人知道，
    因为人人都能碰面时可走的路太多了，挡掉某一条边根本不影响结果，实验问不出东西来。
    真实的小镇是有地理结构的，NPC 每天碰到的就是附近那几个人。

    trust_gate=True 时按信任值决定说不说（跟 agent._pick_share_hint 同一个判断），
    False 时逢人就说——两者的差值就是"信任这一层到底拦住了多少扩散"。"""
    rng = random.Random(seed)
    stores = {i: MemoryStore() for i in range(n)}
    books = {i: RelationshipBook() for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j:
                # 随机给一批人之间埋下不信任；seed 固定，结果可复现
                books[i].apply(f"npc{j}", 0, rng.choice([-3, -3, 0, 1, 2]), "", NOW)
    stores[0].add(RUMOR, 9, NOW)

    def neighbors(i: int) -> list[int]:
        return [(i + d) % n for d in range(-(degree // 2), degree // 2 + 1) if d != 0]

    knows = {0}
    blocked = 0
    for _ in range(rounds):
        for teller in sorted(knows):
            for listener in neighbors(teller):
                if trust_gate and books[teller].trust_of(f"npc{listener}", NOW) < SHARE_TRUST_FLOOR:
                    blocked += 1
                    continue
                if tell(stores[teller], stores[listener], f"npc{teller}", f"npc{listener}", NOW) is not None:
                    knows.add(listener)
    max_hop = max((m.hop for s in stores.values() for m in s.memories), default=0)
    return len(knows), blocked, max_hop


def spread_in_town(n: int, seeds: int, trust_gate: bool, rounds: int = 6, degree: int = 2) -> dict:
    """跑多个种子取平均——单个种子的随机关系配置偶然性太大，跟项目里其他评测脚本
    "多种子聚合"的做法保持一致。"""
    runs = [spread_once(n, s, trust_gate, rounds, degree) for s in range(seeds)]
    return {
        "town_size": n,
        "seeds": seeds,
        "trust_gate": trust_gate,
        "knew_in_the_end": round(mean(r[0] for r in runs), 1),
        "share_of_town": round(mean(r[0] for r in runs) / n, 3),
        "blocked_by_distrust": round(mean(r[1] for r in runs), 1),
        "max_hop": max(r[2] for r in runs),
    }


def render_chain(results: dict[int, list[dict]]) -> str:
    lines = [
        f"衰减系数 HOP_IMPORTANCE_DECAY={HOP_IMPORTANCE_DECAY}，"
        f"值得转述的门槛 SHARE_MIN_IMPORTANCE={SHARE_MIN_IMPORTANCE}，"
        f"听到的话按 IMPORTANCE_HEARD={IMPORTANCE_HEARD} 记",
        "",
        "| 初始重要度 | 一共传了几手 | 各手重要度 | 从第几手起标成传闻 |",
        "| --- | --- | --- | --- |",
    ]
    for origin, steps in sorted(results.items()):
        importances = " → ".join(str(s["importance"]) for s in steps)
        flagged = next((s["hop"] for s in steps if s["flagged_as_hearsay"]), None)
        lines.append(
            f"| {origin} | {len(steps) - 1} | {importances} | {flagged if flagged is not None else '—'} |"
        )
    return "\n".join(lines)


def render_town(rows: list[dict]) -> str:
    lines = [
        f"信任门槛 SHARE_TRUST_FLOOR={SHARE_TRUST_FLOOR}；小镇排成环，每人只跟左右邻居碰面；"
        f"{rows[0]['seeds']} 个种子取平均",
        "",
        "| 小镇人数 | 看不看信任 | 最终知道的人 | 占比 | 被不信任挡下 | 最远传到第几手 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['town_size']} | {'看' if r['trust_gate'] else '不看'} | "
            f"{r['knew_in_the_end']} | {r['share_of_town']:.0%} | {r['blocked_by_distrust']} | {r['max_hop']} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20, help="小镇实验跑几个随机种子取平均")
    ap.add_argument("--town", type=int, default=12)
    args = ap.parse_args()

    chains = {origin: chain(origin) for origin in range(5, 11)}
    town = [spread_in_town(args.town, args.seeds, trust_gate=gate) for gate in (False, True)]

    chain_table = render_chain(chains)
    town_table = render_town(town)
    print(f"\n--- 一条消息能传多远 ---\n\n{chain_table}\n\n--- 信任这一层拦住了多少扩散 ---\n\n{town_table}\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}-gossip-propagation.json").write_text(
        json.dumps({"args": vars(args), "chains": chains, "town": town}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULTS_DIR / f"{stamp}-gossip-propagation.md").write_text(
        f"## 一条消息能传多远\n\n{chain_table}\n\n## 信任这一层拦住了多少扩散\n\n{town_table}\n",
        encoding="utf-8",
    )
    print(f"已保存到 evals/results/{stamp}-gossip-propagation.(json|md)")


if __name__ == "__main__":
    main()
