from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from . import policy
from .protocol import Envelope

app = FastAPI(title="TownMind server")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = Envelope.model_validate_json(raw)
            except ValidationError as e:
                await ws.send_text(
                    Envelope(type="error", payload={"detail": str(e)}).model_dump_json()
                )
                continue

            if msg.type == "hello":
                reply = Envelope(type="welcome", payload={"server": "townmind", "version": "0.1.0"})
            elif msg.type == "observation":
                action = policy.decide(msg.npc_id or "unknown", msg.payload)
                reply = Envelope(type="action", npc_id=msg.npc_id, payload=action)
            else:
                reply = Envelope(type="error", payload={"detail": f"unexpected type {msg.type}"})
            await ws.send_text(reply.model_dump_json())
    except WebSocketDisconnect:
        pass
