from anthropic import AsyncAnthropic

from .base import LLMError, ToolCall

DEFAULT_MODEL = "claude-haiku-4-5"


class AnthropicClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or DEFAULT_MODEL
        self.client = AsyncAnthropic()  # 从环境变量 ANTHROPIC_API_KEY 读取

    async def choose_tool(self, system: str, user: str, tools: list[dict]) -> ToolCall:
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ],
            tool_choice={"type": "any"},  # 必须调用工具
        )
        for block in resp.content:
            if block.type == "tool_use":
                return ToolCall(block.name, dict(block.input))
        raise LLMError("model returned no tool call")
