import asyncio
import logging
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
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
# 几个可选增强，一律默认关、设了对应环境变量才打开：它们要么多花一次 LLM 调用，要么
# 改变 NPC 的行为，关掉时服务的表现跟加这些功能之前完全一致，方便做消融对比。
# 各自的说明见 agent.py 里对应的方法：_rate_importance / _maybe_reflect /
# _maybe_update_relationship / _pick_share_hint / _maybe_compress_conversation。
app.state.agent = Agent(
    _llm,
    memory_dir=DATA_DIR / "memories",
    guard_model=_guard_model,
    embedder=_embedder,
    dynamic_importance=bool(os.getenv("TOWNMIND_DYNAMIC_IMPORTANCE")),
    use_reflection=bool(os.getenv("TOWNMIND_USE_REFLECTION")),
    use_relationships=bool(os.getenv("TOWNMIND_USE_RELATIONSHIPS")),
    use_gossip=bool(os.getenv("TOWNMIND_USE_GOSSIP")),
    compress_conversations=bool(os.getenv("TOWNMIND_COMPRESS_CONVERSATIONS")),
    # 这两个真实数据没测出正向收益（见 README"CoVe 式重试和 Reflexion 式教训记忆"
    # 小节），保持默认关。这里照样接上环境变量，是为了想重新验证时不用改代码——
    # 接上不等于打开，不填就是关的。
    verify_and_revise=bool(os.getenv("TOWNMIND_VERIFY_AND_REVISE")),
    reflexion_lessons=bool(os.getenv("TOWNMIND_REFLEXION_LESSONS")),
)


# 网页 demo 挂在根路径上：服务一起来，浏览器打开 http://127.0.0.1:8000/ 就能玩，
# 不用再手动去文件系统里双击 html。真正的 mount 放在本文件末尾——StaticFiles 挂在 "/"
# 上会吃掉所有未匹配的路径，必须等 /health /stats /rumor 这些路由都注册完再挂，
# 否则它们会被静态文件服务盖掉。
_WEB_DEMO = Path(__file__).resolve().parents[1] / "web_demo"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/memories/{npc_id}")
async def memories(npc_id: str) -> list[dict]:
    """查看某个 NPC 现在记得什么（最新的 50 条），方便调试和理解记忆系统。"""
    return app.state.agent.memory_dump(npc_id)


@app.get("/rumor/{topic}")
async def rumor(topic: str) -> dict:
    """一条消息现在传到哪儿了：谁知道、第几手、各人嘴里是什么版本、从谁那儿听来的。
    传话 demo 的数据源——把"信息在小镇里怎么流动"这件本来只存在于 JSON 里的事画出来。"""
    return app.state.agent.rumor_trace(topic)


@app.get("/recall/{npc_id}")
async def recall(npc_id: str) -> dict:
    """某个 NPC 最近一次决策时想起了什么，以及每条记忆的三项分数各是多少。

    这是整个记忆系统里最值得看、却一直看不见的部分：光看 NPC 说了什么，看不出它
    凭什么想起这条而不是那条。recall_explained() 早就写好了，这里把它接出来。"""
    return {"npc_id": npc_id, "recalled": app.state.agent.last_recall.get(npc_id, [])}


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
                # 玩家说话：先过入口检查，再变成附近 NPC 能听到的说话事件；不需要回复。
                # 这一步之前完全没有日志——玩家打字发出去之后，服务端这边收没收到、
                # 入口检查有没有拦下，从终端输出上完全看不出来，调试的时候只能瞎猜。
                text = str(msg.payload.get("text", ""))
                res = app.state.agent.hear_player(text, msg.payload.get("pos"))
                if res.ok:
                    log.info("[player] said: %s%s", res.text, f" flags={list(res.flags)}" if res.flags else "")
                else:
                    log.info("[player] said 被入口检查拦下（flags=%s）：%s", list(res.flags), text)
            elif msg.type == "observation":
                task = asyncio.create_task(handle_observation(msg))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
            elif msg.type == "whisper":
                # 玩家凑到某个 NPC 耳边说一件事：只有他知道，旁边的人听不见。
                # 跟 player_say 的区别就在这儿——那个是当众说，会变成附近所有人都能
                # 听到的说话事件；这个是传话游戏的起点，必须只有一个源头。
                info = app.state.agent.whisper(
                    msg.npc_id or "unknown",
                    str(msg.payload.get("text", "")),
                    str(msg.payload.get("topic", "rumor")),
                )
                log.info("[player] 悄悄告诉 %s：%s", info["npc_id"], info["text"])
            elif msg.type == "status_query":
                await send(Envelope(type="status", npc_id=msg.npc_id, payload=_companion_status(msg.npc_id or "unknown")))
            else:
                await send(Envelope(type="error", payload={"detail": f"unexpected type {msg.type}"}))
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()


if _WEB_DEMO.is_dir():
    app.mount("/", StaticFiles(directory=str(_WEB_DEMO), html=True), name="web_demo")
