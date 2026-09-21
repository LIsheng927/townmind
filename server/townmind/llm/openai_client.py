import json

from openai import AsyncOpenAI

from .base import LLMError, ToolCall

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or DEFAULT_MODEL
        self.client = AsyncOpenAI()  # 从环境变量 OPENAI_API_KEY 读取

    async def choose_tool(self, system: str, user: str, tools: list[dict]) -> ToolCall:
        resp = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]},
                }
                for t in tools
            ],
            tool_choice="required",
        )
        calls = resp.choices[0].message.tool_calls
        if not calls:
            raise LLMError("model returned no tool call")
        return ToolCall(calls[0].function.name, json.loads(calls[0].function.arguments))
