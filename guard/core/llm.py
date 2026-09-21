"""数据生成阶段用的、轻量的大模型调用（不依赖 server 的工具调用框架，只要"给提示词、拿文本"）。

复用 server/.env 里的 provider 和 key（guard 和 server 用的是同一个账号/预算），
是唯一一处和 server 有依赖关系的地方，为的是不用重复配置 API key。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / "server" / ".env")

_PROVIDER = (os.getenv("TOWNMIND_LLM_PROVIDER") or "").lower()


def _client_and_model():
    if _PROVIDER == "openai":
        from openai import OpenAI

        return OpenAI(), os.getenv("TOWNMIND_MODEL") or "gpt-4o-mini"
    if _PROVIDER == "anthropic":
        import anthropic

        return anthropic.Anthropic(), os.getenv("TOWNMIND_MODEL") or "claude-haiku-4-5"
    raise SystemExit(
        "没有可用的大模型：请检查 server/.env 里 TOWNMIND_LLM_PROVIDER 和对应的 key 是否匹配。"
    )


def generate_text(system: str, user: str, temperature: float = 1.0) -> str:
    """给一段系统提示词和用户提示词，拿回纯文本。用于批量造数据，不涉及工具调用。"""
    client, model = _client_and_model()
    if _PROVIDER == "openai":
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")
