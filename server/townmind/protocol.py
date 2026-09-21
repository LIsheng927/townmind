"""Unity <-> server 通信协议。所有消息都是一个 JSON 信封。"""
from typing import Any, Literal

from pydantic import BaseModel, Field


class Envelope(BaseModel):
    type: Literal["hello", "welcome", "observation", "action", "error"]
    npc_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
