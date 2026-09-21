"""NPC 决策策略。

第 1 步：临时的随机游走，只为验证 Unity <-> 服务端闭环。
第 2 步会替换成 LLM 驱动的决策，届时保持同样的接口：decide(npc_id, observation) -> action。
"""
import random

WORLD_HALF_SIZE = 8.0


def decide(npc_id: str, observation: dict, rng: random.Random | None = None) -> dict:
    rng = rng or random
    return {
        "name": "move_to",
        "x": round(rng.uniform(-WORLD_HALF_SIZE, WORLD_HALF_SIZE), 2),
        "z": round(rng.uniform(-WORLD_HALF_SIZE, WORLD_HALF_SIZE), 2),
    }
