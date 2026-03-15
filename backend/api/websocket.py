"""Project websocket routes."""
from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()


class ConnectionManager:
    HEARTBEAT_TIMEOUT = 60.0

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}
        self._global_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket, project_id: Optional[str] = None):
        await websocket.accept()
        if project_id:
            self._connections.setdefault(project_id, []).append(websocket)
            logger.info(f"WebSocket connected for project {project_id}")
        else:
            self._global_connections.append(websocket)
            logger.info('Global WebSocket connected')

    def disconnect(self, websocket: WebSocket, project_id: Optional[str] = None):
        if project_id:
            connections = self._connections.get(project_id, [])
            if websocket in connections:
                connections.remove(websocket)
            if not connections and project_id in self._connections:
                self._connections.pop(project_id, None)
            logger.info(f"WebSocket disconnected for project {project_id}")
            return

        if websocket in self._global_connections:
            self._global_connections.remove(websocket)
            logger.info('Global WebSocket disconnected')

    async def send_message(self, websocket: WebSocket, message: dict):
        try:
            await websocket.send_json(message)
        except Exception as exc:
            logger.warning(f"WebSocket send failed: {exc}")

    async def broadcast_to_project(self, project_id: str, message: dict):
        for websocket in list(self._connections.get(project_id, [])):
            await self.send_message(websocket, message)
        for websocket in list(self._global_connections):
            await self.send_message(websocket, {**message, 'project_id': project_id})

    async def broadcast_global(self, message: dict):
        for sockets in self._connections.values():
            for websocket in list(sockets):
                await self.send_message(websocket, message)
        for websocket in list(self._global_connections):
            await self.send_message(websocket, message)


manager = ConnectionManager()


@router.websocket('/project/{project_id}')
async def websocket_project(websocket: WebSocket, project_id: str):
    await manager.connect(websocket, project_id)
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=manager.HEARTBEAT_TIMEOUT)
            except asyncio.TimeoutError:
                await manager.send_message(websocket, {'type': 'ping'})
                continue

            message_type = data.get('type')
            if message_type == 'ping':
                await manager.send_message(websocket, {'type': 'pong'})
            elif message_type == 'get_progress':
                from api.routes.project import load_project

                project = load_project(project_id)
                if project:
                    await manager.send_message(
                        websocket,
                        {
                            'type': 'progress',
                            'stage': project.progress.stage,
                            'progress': project.progress.progress,
                            'message': project.progress.message,
                        },
                    )
    except WebSocketDisconnect:
        manager.disconnect(websocket, project_id)
    except Exception as exc:
        logger.error(f"Project websocket error for {project_id}: {exc}")
        manager.disconnect(websocket, project_id)


@router.websocket('/global')
async def websocket_global(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=manager.HEARTBEAT_TIMEOUT)
            except asyncio.TimeoutError:
                await manager.send_message(websocket, {'type': 'ping'})
                continue

            if data.get('type') == 'ping':
                await manager.send_message(websocket, {'type': 'pong'})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:
        logger.error(f"Global websocket error: {exc}")
        manager.disconnect(websocket)
