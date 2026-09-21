import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .base import LLMClient

log = logging.getLogger("townmind.llm")
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"  # server/.env


def make_client() -> LLMClient | None:
    """按环境变量创建 LLM 客户端；没配置 key 时返回 None（NPC 走规则兜底）。"""
    load_dotenv(ENV_PATH)
    provider = os.getenv("TOWNMIND_LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("TOWNMIND_MODEL") or None
    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        from .anthropic_client import AnthropicClient

        return AnthropicClient(model)
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        from .openai_client import OpenAIClient

        return OpenAIClient(model)
    log.warning("no LLM configured (provider=%s); NPCs will use the rule-based fallback", provider)
    return None
