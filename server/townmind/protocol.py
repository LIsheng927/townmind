"""Unity <-> server 通信协议。所有消息都是一个 JSON 信封。"""
from typing import Any, Literal

from pydantic import BaseModel, Field


class Envelope(BaseModel):
    type: Literal[
        "hello", "welcome", "observation", "position", "player_say", "action", "error",
        "whisper",  # 玩家私下告诉某一个 NPC 一件事（旁边的人听不见）。传话 demo 用：
        # 从单一源头出发，才看得清消息是怎么一跳一跳扩散出去的
        "status_query", "status",  # 查询某个 NPC 当前的伙伴状态（跟没跟着、手上拿没拿着、任务是什么）——
        # 目前只有网页版小demo在用，用来把"手上拿没拿着东西"这类只存在于服务端内存里的
        # 事实显示出来，不用靠猜或者解析台词文本
    ]
    npc_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
