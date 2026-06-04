import importlib.util
import sys
import types
from pathlib import Path

import pytest


API_ROOT = Path(__file__).resolve().parents[1]


def _clear_memobase_modules() -> None:
    for name in list(sys.modules):
        if name == "memobase_server" or name.startswith("memobase_server."):
            del sys.modules[name]


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _BufferStatus:
    idle = "idle"
    processing = "processing"


class _BlobType:
    chat = "chat"

    def __str__(self):
        return self.chat


class _TraceLog:
    def debug(self, *_args, **_kwargs):
        pass

    def info(self, *_args, **_kwargs):
        pass

    def warning(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


def _load_buffer_background():
    _clear_memobase_modules()
    _package("memobase_server")
    controllers_pkg = _package("memobase_server.controllers")
    models_pkg = _package("memobase_server.models")

    _module(
        "memobase_server.env",
        BufferStatus=_BufferStatus,
        TRACE_LOG=_TraceLog(),
    )
    database_module = _module(
        "memobase_server.models.database",
        BufferZone=types.SimpleNamespace(),
    )
    blob_module = _module("memobase_server.models.blob", BlobType=_BlobType)
    models_pkg.database = database_module
    models_pkg.blob = blob_module
    _module(
        "memobase_server.connectors",
        Session=lambda: None,
        PROJECT_ID="test-project",
        get_redis_client=lambda: None,
    )
    _module("memobase_server.controllers.modal", BLOBS_PROCESS={_BlobType.chat: object()})
    _module("memobase_server.controllers.buffer", flush_buffer_by_ids=lambda *args, **kwargs: None)

    spec = importlib.util.spec_from_file_location(
        "memobase_server.controllers.buffer_background",
        API_ROOT / "memobase_server" / "controllers" / "buffer_background.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    controllers_pkg.buffer_background = module
    spec.loader.exec_module(module)
    return module


buffer_background = _load_buffer_background()
BufferStatus = _BufferStatus
BlobType = _BlobType


class FakeRedis:
    def __init__(self, items=None):
        self.items = list(items or [])

    async def llen(self, _key):
        return len(self.items)

    async def rpush(self, _key, value):
        self.items.append(value)
        return len(self.items)


@pytest.mark.asyncio
async def test_requeue_processing_buffers_when_queue_is_empty(monkeypatch):
    fake_redis = FakeRedis()

    def fake_get_buffer_ids_by_status(user_id, project_id, blob_type, status):
        assert user_id == "user-1"
        assert project_id == "project-1"
        assert blob_type == BlobType.chat
        assert status == BufferStatus.processing
        return ["buf-1", "buf-2", "buf-3"]

    monkeypatch.setattr(
        buffer_background,
        "_get_buffer_ids_by_status",
        fake_get_buffer_ids_by_status,
    )

    recovered = await buffer_background._enqueue_processing_buffers_if_queue_empty(
        fake_redis,
        "queue-key",
        "user-1",
        "project-1",
        BlobType.chat,
        batch_size=2,
    )

    assert recovered == 3
    assert fake_redis.items == ["buf-1::buf-2", "buf-3"]


@pytest.mark.asyncio
async def test_requeue_processing_buffers_preserves_existing_queue(monkeypatch):
    fake_redis = FakeRedis(["queued-buf"])

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("processing buffers should not be queried")

    monkeypatch.setattr(
        buffer_background,
        "_get_buffer_ids_by_status",
        fail_if_called,
    )

    recovered = await buffer_background._enqueue_processing_buffers_if_queue_empty(
        fake_redis,
        "queue-key",
        "user-1",
        "project-1",
        BlobType.chat,
        batch_size=2,
    )

    assert recovered == 0
    assert fake_redis.items == ["queued-buf"]
