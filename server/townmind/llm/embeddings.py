"""语义检索用的 embedding 客户端：把一段文字变成一个向量，方便用余弦相似度衡量"相关度"
（见 townmind/memory.py 里 MemoryStore.score 的说明）。

跟 anthropic_client.py / openai_client.py 是同一个套路——make_embedder() 是这里的
factory：没配 OPENAI_API_KEY 时返回 None，上层（Agent._remember）据此优雅跳过，记忆
系统自动退化成不含语义相关度的旧公式，不影响服务启动。

聊天用哪个大模型（TOWNMIND_LLM_PROVIDER=anthropic/openai）和做 embedding 用哪个，是两件
独立的事：Anthropic 目前不提供 embedding 接口，所以这里固定走 OpenAI；即使聊天走的是
Claude，只要另外配了 OPENAI_API_KEY，语义检索照样能用。
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger("townmind.embeddings")

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"  # server/.env，跟 llm/factory.py 是同一个文件
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


class OpenAIEmbedder:
    def __init__(self, model: str | None = None) -> None:
        from openai import AsyncOpenAI  # 延迟导入，跟 llm/factory.py 里两个 provider client 的写法一致：

        # 没真的用到这个 provider 时，不强求 openai 这个包在当前环境里一定装得上
        self.model = model or DEFAULT_EMBED_MODEL
        self.client = AsyncOpenAI()  # 从环境变量 OPENAI_API_KEY 读取

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """一次性把好几段文字都变成向量（批量调用，省一次决策周期里的好几次网络往返）。
        texts 为空时直接返回空列表，不发请求。"""
        if not texts:
            return []
        resp = await self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


def make_embedder() -> "OpenAIEmbedder | None":
    """按环境变量创建 embedding 客户端；没配 key 时返回 None（记忆退化成不带语义相关度的旧公式）。

    自己读一遍 .env（不依赖调用方是不是先调用过 make_client()）——之前这里漏了这一步，
    main.py 里能用纯粹是因为 make_client() 先跑了一遍、顺带把 .env 读进了进程环境变量；
    单独只用 embedder 的脚本（比如 evals/memory_recall.py）跳过 make_client() 时就会读不到 key。
    """
    load_dotenv(ENV_PATH)
    if not os.getenv("OPENAI_API_KEY"):
        log.warning("no OPENAI_API_KEY; memory recall falls back to the non-semantic relevance term")
        return None
    return OpenAIEmbedder()
