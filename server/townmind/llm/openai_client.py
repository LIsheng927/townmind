import asyncio
import json
import re

from openai import AsyncOpenAI, RateLimitError

from .base import LLMError, ToolCall

DEFAULT_MODEL = "gpt-4o-mini"
RETRIES = 3


class OpenAIClient:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or DEFAULT_MODEL
        self.client = AsyncOpenAI()  # 从环境变量 OPENAI_API_KEY 读取
        self.rate_limit_retries = 0  # 观察用：撞了几次 429 又重试成功的

    async def _create_with_backoff(self, **kwargs):
        """429（每分钟 token 上限）按服务端给的等待时间退避重试，最多 RETRIES 次。

        无头评测用虚拟时钟，5 分钟的小镇几十秒跑完，请求全挤在一起，真实跑过一次就撞上了
        （TPM 200k，"Please try again in 163ms"）；真实服务里 NPC 一多也会撞。不重试的话
        这一轮直接退回行为树台词，评测里的重复率会被固定台词污染。等待时间优先用报错里
        的数字，没有就 0.5s 起翻倍；三次都不行才把异常抛出去让上层熔断/兜底。"""
        delay = 0.5
        for attempt in range(RETRIES + 1):
            try:
                return await self.client.chat.completions.create(**kwargs)
            except RateLimitError as e:
                if attempt == RETRIES:
                    raise
                m = re.search(r"try again in (\d+)(ms|s)", str(e))
                wait = (int(m.group(1)) / (1000 if m.group(2) == "ms" else 1)) if m else delay
                await asyncio.sleep(min(max(wait, 0.05), 5.0) + 0.05)
                delay *= 2
                self.rate_limit_retries += 1

    async def choose_tool(self, system: str, user: str, tools: list[dict]) -> ToolCall:
        resp = await self._create_with_backoff(
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
        usage = resp.usage
        return ToolCall(
            calls[0].function.name,
            json.loads(calls[0].function.arguments),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )
