"""最小的行为树（Behavior Tree）：游戏 AI 里最常用的"规则式决策"结构。

树由节点组成，每个节点每次"跳动"（tick）返回成功或失败：
  - Selector（选择）：从左到右依次尝试子节点，第一个成功的就停下并成功；全部失败才失败。像"优先做 A，不行就做 B，再不行做 C"。
  - Sequence（序列）：从左到右依次执行，全部成功才成功；中途有一个失败就整体失败。像"条件都满足了，才动手"。
  - Condition（条件）：判断某件事是否成立。
  - Action（动作）：真正决定 NPC 做什么，把结果写进上下文。
"""
from enum import Enum
from typing import Callable


class Status(Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class Node:
    def tick(self, ctx) -> Status:  # pragma: no cover - 接口
        raise NotImplementedError


class Selector(Node):
    def __init__(self, *children: Node) -> None:
        self.children = children

    def tick(self, ctx) -> Status:
        for c in self.children:
            if c.tick(ctx) is Status.SUCCESS:
                return Status.SUCCESS
        return Status.FAILURE


class Sequence(Node):
    def __init__(self, *children: Node) -> None:
        self.children = children

    def tick(self, ctx) -> Status:
        for c in self.children:
            if c.tick(ctx) is Status.FAILURE:
                return Status.FAILURE
        return Status.SUCCESS


class Condition(Node):
    def __init__(self, fn: Callable[[object], bool]) -> None:
        self.fn = fn

    def tick(self, ctx) -> Status:
        return Status.SUCCESS if self.fn(ctx) else Status.FAILURE


class Action(Node):
    """fn(ctx) 返回一个动作字典（写进 ctx.action 并成功），返回 None 则失败。"""

    def __init__(self, fn: Callable[[object], dict | None]) -> None:
        self.fn = fn

    def tick(self, ctx) -> Status:
        action = self.fn(ctx)
        if action is None:
            return Status.FAILURE
        ctx.action = action
        return Status.SUCCESS
