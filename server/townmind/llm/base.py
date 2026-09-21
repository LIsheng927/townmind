"""LLM 接入层：统一接口，屏蔽不同厂商 SDK 的差异。"""
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(Exception):
    pass


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    async def choose_tool(self, system: str, user: str, tools: list[dict]) -> ToolCall:
        """让模型在给定工具中必选其一。tools 为厂商无关格式：
        {"name": str, "description": str, "parameters": <JSON Schema>}"""
        ...
