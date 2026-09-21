"""无头仿真：不需要 Unity，直接在 Python 里模拟 NPC 的"身体"，驱动 Agent 跑一段时间。

身体的行为刻意和 Unity 的 NpcController 保持一致：
  - 只有停下来时才向大脑上报位置、请求下一步；
  - 收到动作后至少等 1 秒；说完话等 3 秒；休息就等它指定的秒数；
  - 走路的速度是 2.5 米/秒；
  - 无论走路还是停下，位置都在持续上报（Unity 里每 0.5 秒一次，这里每个仿真步一次）。
时间是"虚拟时间"：一秒的仿真不需要真等一秒，所以几分钟的场景几十秒就能跑完。
（局限：仿真里大模型的响应延迟不占虚拟时间，延迟单独统计。）
"""
import asyncio
import math
from dataclasses import dataclass

START_POSITIONS = {"alice": (-3.0, 0.0), "bob": (0.0, 3.0), "carol": (3.0, -2.0)}  # 和 Unity 里一致
SPEED = 2.5


class SimClock:
    """可手动推进的时钟，注入给 Agent，让评测不依赖真实时间、结果可复现。"""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t


@dataclass
class Body:
    npc_id: str
    x: float
    z: float
    target: tuple[float, float] | None = None
    next_report: float = 0.0


def _step(b: Body, dt: float) -> None:
    if b.target is None:
        return
    dx, dz = b.target[0] - b.x, b.target[1] - b.z
    dist = math.hypot(dx, dz)
    if dist <= SPEED * dt:
        b.x, b.z = b.target
        b.target = None
    else:
        b.x += dx / dist * SPEED * dt
        b.z += dz / dist * SPEED * dt


def _apply(b: Body, action: dict, now: float) -> None:
    b.next_report = now + 1.0
    if action["name"] == "move_to":
        b.target = (action["x"], action["z"])
    elif action["name"] == "say":
        b.next_report = now + 3.0
    elif action["name"] == "idle":
        b.next_report = now + action["seconds"]


async def simulate(agent, clock: SimClock, seconds: float = 300.0, dt: float = 1.0) -> None:
    bodies = [Body(n, x, z, next_report=clock.t) for n, (x, z) in START_POSITIONS.items()]
    end = clock.t + seconds
    while clock.t < end:
        for b in bodies:
            _step(b, dt)
        for b in bodies:  # 和 Unity 一致：走路时也在持续上报位置
            agent.update_position(b.npc_id, [b.x, b.z])
        ready = [b for b in bodies if b.target is None and clock.t >= b.next_report]
        # 同一时刻需要决策的 NPC 并发去问大脑，和真实服务里的并发一致
        actions = await asyncio.gather(*(agent.decide(b.npc_id, {"pos": [b.x, b.z]}) for b in ready))
        for b, a in zip(ready, actions):
            _apply(b, a, clock.t)
        clock.t += dt
