import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from . import world
from .agent import Agent
from .llm.factory import make_client
from .protocol import Envelope

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("townmind")

app = FastAPI(title="TownMind server")
_llm = make_client()  # 先创建（这一步会读取 .env），再读数据目录配置
DATA_DIR = Path(os.getenv("TOWNMIND_DATA_DIR") or Path(__file__).resolve().parents[1] / "data")
app.state.agent = Agent(_llm, memory_dir=DATA_DIR / "memories")


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
    return dict(app.state.agent.stats)


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
                await send(Envelope(type="welcome", payload={"server": "townmind", "version": "0.3.0", "locations": world.locations_payload()}))
            elif msg.type == "position":
                # 轻量的位置更新：不触发决策，也没有回复
                app.state.agent.update_position(msg.npc_id or "unknown", msg.payload.get("pos"))
            elif msg.type == "observation":
                task = asyncio.create_task(handle_observation(msg))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            else:
                await send(Envelope(type="error", payload={"detail": f"unexpected type {msg.type}"}))
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()
