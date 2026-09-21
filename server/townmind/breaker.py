"""熔断器：大模型服务连续出问题时，暂时不再请求它，直接走兜底，过一阵再试探恢复。

三种状态（像家里的保险丝）：
  - closed（正常）：请求照常发出。连续失败达到 threshold 次，就"跳闸"。
  - open（跳闸）：一段时间（cooldown 秒）内一律不发请求，NPC 直接用行为树。省钱，也不让玩家等超时。
  - half_open（试探）：冷却结束后，只放一个请求过去试试。成功 -> 回到 closed；失败 -> 重新跳闸。
"""
from typing import Callable


class CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown: float = 30.0, clock: Callable[[], float] | None = None) -> None:
        import time

        self.threshold = threshold
        self.cooldown = cooldown
        self.clock = clock or time.time
        self.failures = 0
        self.opened_at: float | None = None
        self.probing = False  # half_open 时是否已经有一个试探请求在路上

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if self.clock() - self.opened_at >= self.cooldown:
            return "half_open"
        return "open"

    def allow(self) -> bool:
        s = self.state
        if s == "closed":
            return True
        if s == "half_open" and not self.probing:
            self.probing = True  # 只放一个试探请求
            return True
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None
        self.probing = False

    def record_failure(self) -> None:
        self.probing = False
        self.failures += 1
        if self.opened_at is not None or self.failures >= self.threshold:
            self.opened_at = self.clock()  # 试探失败，或连续失败太多：（重新）跳闸
