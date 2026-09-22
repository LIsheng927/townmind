import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from . import world
from .agent import Agent
from .guard_model import ENV_ENABLE, GuardModel
from .llm.embeddings import make_embedder
from .llm.factory import make_client
from .protocol import Envelope

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("townmind")

app = FastAPI(title="TownMind server")
_llm = make_client()  # 先创建（这一步会读取 .env），再读数据目录配置
DATA_DIR = Path(os.getenv("TOWNMIND_DATA_DIR") or Path(__file__).resolve().parents[1] / "data")
# 自研防御模型是可选的第二层安全检查（见 guard/README.md "接入服务"一节）：没设这个环境变量、
# 没装可选依赖（uv sync --group guard-model）、或者没有训练好的 adapter，都会在真正用到时
# 优雅跳过，不影响服务启动，所以这里可以放心地无条件构造 GuardModel()。
_guard_model = GuardModel() if os.getenv(ENV_ENABLE) else None
# 语义记忆检索的 embedding 客户端：同样是可选的，没配 OPENAI_API_KEY 时 make_embedder()
# 返回 None，记忆的"相关度"评分自动退化成旧版的"认不认人"，不影响服务启动（见 llm/embeddings.py）。
_embedder = make_embedder()
# 动态重要度打分、反思：都会多花一次 LLM 调用，默认关，设了对应环境变量才打开
# （见 agent.py 里 _rate_importance/_maybe_reflect 的说明）。
app.state.agent = Agent(
    _llm,
    memory_dir=DATA_DIR / "memories",
    guard_model=_guard_model,
    embedder=_embedder,
    dynamic_importance=bool(os.getenv("TOWNMIND_DYNAMIC_IMPORTANCE")),
    use_reflection=bool(os.getenv("TOWNMIND_USE_REFLECTION")),
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/memories/{npc_id}")
async def memories(npc_id: str) -> list[dict]:
    """查看某个 NPC 现在记得什么（最新的 50 条），方便调试和理解记忆系统。"""
    return app.state.agent.memory_dump(npc_id)


@app.get("/stats")
async def stats() -> dict:
    """决策统计：大模型调用次数、失败次数、规则决策次数。用于观察成本。"""
    return {**app.state.agent.stats, "breaker": app.state.agent.breaker.state}


def _companion_status(npc_id: str) -> dict:
    """伙伴状态：跟没跟着、手上拿没拿着东西、任务是什么——这些只存在于服务端内存里的
    "物理事实"，网页版小demo靠这个接口把它们显示出来，不用靠解析台词文本去猜。"""
    agent = app.state.agent
    task = agent.tasks.get(npc_id)
    return {
        "npc_id": npc_id,
        "pos": list(agent.positions.get(npc_id, (0.0, 0.0))),
        "following": npc_id in agent.following,
        "holding": agent.holding.get(npc_id),
        "task": {"item": task.item, "destination": task.destination} if task else None,
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    send_lock = asyncio.Lock()  # 多个并发任务共用一个连接，发送要串行
    tasks: set[asyncio.Task] = set()

    async def send(env: Envelope) -> None:
        async with send_lock:
            await ws.send_text(env.model_dump_json())

    async def handle_observation(msg: Envelope) -> None:
        # 每个 NPC 的决策独立成任务：某个 NPC 的 LLM 调用慢，不会阻塞其他 NPC
        try:
            action = await app.state.agent.decide(msg.npc_id or "unknown", msg.payload)
            await send(Envelope(type="action", npc_id=msg.npc_id, payload=action))
        except Exception:
            log.exception("failed to handle observation")

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = Envelope.model_validate_json(raw)
            except ValidationError as e:
                await send(Envelope(type="error", payload={"detail": str(e)}))
                continue

            if msg.type == "hello":
                await send(
                    Envelope(
                        type="welcome",
                        payload={
                            "server": "townmind",
                            "version": "0.3.0",
                            "locations": world.locations_payload(),
                            "items": world.items_payload(),
                        },
                    )
                )
            elif msg.type == "position":
                # 轻量的位置更新：不触发决策，也没有回复
                app.state.agent.update_position(msg.npc_id or "unknown", msg.payload.get("pos"))
            elif msg.type == "player_say":
                # 玩家说话：先过入口检查，再变成附近 NPC 能听到的说话事件；不需要回复
                app.state.agent.hear_player(str(msg.payload.get("text", "")), msg.payload.get("pos"))
            elif msg.type == "observation":
                task = asyncio.create_task(handle_observation(msg))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            elif msg.type == "status_query":
                await send(Envelope(type="status", npc_id=msg.npc_id, payload=_companion_status(msg.npc_id or "unknown")))
            else:
                await send(Envelope(type="error", payload={"detail": f"unexpected type {msg.type}"}))
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()
