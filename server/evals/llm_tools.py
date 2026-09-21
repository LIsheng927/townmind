import time
import zlib

from townmind.agent import PLACE_NAMES
from townmind.llm.base import ToolCall


class MeteredLLM:
    """包一层，记录每次调用的耗时，用来算延迟的 P50/P95。"""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.latencies: list[float] = []

    async def choose_tool(self, system, user, tools):
        t0 = time.perf_counter()
        try:
            return await self.inner.choose_tool(system, user, tools)
        finally:
            self.latencies.append(time.perf_counter() - t0)


_LINES = (
    "早上好！今天法棍刚出炉，来一个吗？",
    "面粉又涨价了，真让人发愁。",
    "听说周六广场有集市，你去吗？",
    "铁匠铺最近缺铁矿，Bob 一直在等。",
    "你好呀，最近怎么样？",
    "我在打铁，别打扰我。",
    "我刚到这个小镇，还不太熟悉。",
    "小镇的面包节快到了，你期待吗？",  # 故意放一句"编造"的话，让编造率指标有东西可测
)


class OfflineLLM:
    """确定性的假大模型：不联网、不花钱，只用来测试评测流程本身。

    ⚠ 它的输出是按提示词哈希挑的固定句子，不代表任何真实模型的水平，
    用它跑出来的指标数字没有参考意义。
    """

    async def choose_tool(self, system, user, tools):
        h = zlib.crc32(user.encode("utf-8"))
        usage = {"input_tokens": len(user) // 2, "output_tokens": 20}
        if "已经说了 3 句话" in user or h % 7 == 0:
            return ToolCall("end_conversation", {"farewell": "我先去忙了，再见！"}, **usage)
        if "你刚听到" in user or h % 3 != 0:
            return ToolCall("say", {"text": _LINES[h % len(_LINES)]}, **usage)
        return ToolCall("go_to", {"place": PLACE_NAMES[h % len(PLACE_NAMES)]}, **usage)
