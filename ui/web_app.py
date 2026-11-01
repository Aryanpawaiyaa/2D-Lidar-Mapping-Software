"""Real-time Web UI for 2D Lidar SLAM visualization using FastAPI, WebSockets, and HTML5 Canvas."""

import asyncio
import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from slam.system import LidarSlamSystem

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def create_web_app(slam_system: LidarSlamSystem) -> FastAPI:
    """Create and configure FastAPI application for SLAM streaming."""
    app = FastAPI(title="Persistent 2D Lidar SLAM - YDLIDAR TG50")
    os.makedirs(STATIC_DIR, exist_ok=True)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    connected_clients: List[WebSocket] = []

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        index_file = os.path.join(STATIC_DIR, "index.html")
        return FileResponse(index_file)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        connected_clients.append(websocket)
        logger.info("Web UI client connected.")
        try:
            while True:
                # Listen for commands from UI
                msg_text = await websocket.receive_text()
                cmd = json.loads(msg_text)
                action = cmd.get("action")

                if action == "save":
                    map_name = cmd.get("name", "manual_save")
                    saved_path = slam_system.save_map(map_name)
                    await websocket.send_json({"event": "saved", "path": saved_path, "name": map_name})

                elif action == "load":
                    map_name = cmd.get("name", "current_session")
                    try:
                        slam_system.load_map(map_name)
                        await websocket.send_json({"event": "loaded", "name": map_name})
                    except Exception as e:
                        await websocket.send_json({"event": "error", "message": str(e)})

                elif action == "export":
                    map_name = cmd.get("name", "current_session")
                    saved_path = slam_system.save_map(map_name)
                    await websocket.send_json({"event": "exported", "name": map_name, "path": saved_path})

        except WebSocketDisconnect:
            if websocket in connected_clients:
                connected_clients.remove(websocket)
            logger.info("Web UI client disconnected.")
        except Exception as e:
            if websocket in connected_clients:
                connected_clients.remove(websocket)

    async def broadcast_telemetry(payload: Dict[str, Any]):
        for client in list(connected_clients):
            try:
                await client.send_json(payload)
            except Exception:
                if client in connected_clients:
                    connected_clients.remove(client)

    app.state.broadcast_telemetry = broadcast_telemetry
    return app

