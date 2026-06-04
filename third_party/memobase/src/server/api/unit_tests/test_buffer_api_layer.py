import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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


class _BaseResponse:
    def __init__(self, data=None, errno=0, errmsg=""):
        self.data = data
        self.errno = errno
        self.errmsg = errmsg


class _ChatModalAPIResponse(_BaseResponse):
    pass


class _IdsResponse(_BaseResponse):
    pass


@dataclass
class _IdsData:
    ids: list[str]


@dataclass
class _Promise:
    value: object
    ok_value: bool = True

    @classmethod
    def resolve(cls, value):
        return cls(value)

    def ok(self):
        return self.ok_value

    def data(self):
        return self.value

    def to_response(self, response_cls):
        return response_cls(data=self.value, errno=0, errmsg="")


def _load_buffer_api():
    _clear_memobase_modules()
    _package("memobase_server")
    _package("memobase_server.api_layer")
    controllers_pkg = _package("memobase_server.controllers")
    models_pkg = _package("memobase_server.models")

    response_module = _module(
        "memobase_server.models.response",
        UUID=str,
        IdsData=_IdsData,
        IdsResponse=_IdsResponse,
        BaseResponse=_BaseResponse,
        ChatModalAPIResponse=_ChatModalAPIResponse,
    )
    blob_module = _module("memobase_server.models.blob", BlobType=_BlobType)
    utils_module = _module("memobase_server.models.utils", Promise=_Promise)
    models_pkg.response = response_module
    models_pkg.blob = blob_module
    models_pkg.utils = utils_module

    buffer_controller = types.SimpleNamespace(get_unprocessed_buffer_ids=None)
    buffer_background = types.SimpleNamespace(
        BufferStatus=_BufferStatus,
        flush_buffer_background_running=lambda *args, **kwargs: None,
        flush_buffer_by_ids_in_background=lambda *args, **kwargs: None,
    )
    full_module = _module(
        "memobase_server.controllers.full",
        buffer=buffer_controller,
        buffer_background=buffer_background,
    )
    controllers_pkg.full = full_module

    spec = importlib.util.spec_from_file_location(
        "memobase_server.api_layer.buffer",
        API_ROOT / "memobase_server" / "api_layer" / "buffer.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


buffer_api = _load_buffer_api()
BufferStatus = _BufferStatus
IdsData = _IdsData
Promise = _Promise


@pytest.mark.asyncio
async def test_flush_buffer_restarts_background_for_processing_only_user(monkeypatch):
    calls = []

    async def fake_get_unprocessed_buffer_ids(
        user_id, project_id, buffer_type, select_status=BufferStatus.idle
    ):
        if select_status == BufferStatus.idle:
            return Promise.resolve(IdsData(ids=[]))
        if select_status == BufferStatus.processing:
            return Promise.resolve(IdsData(ids=["550e8400-e29b-41d4-a716-446655440000"]))
        raise AssertionError(f"unexpected status: {select_status}")

    async def fake_flush_buffer_background_running(user_id, project_id, buffer_type):
        calls.append((user_id, project_id, buffer_type))

    class FakeBackgroundTasks:
        def add_task(self, func, *args):
            calls.append((func.__name__, *args))

    monkeypatch.setattr(
        buffer_api.controllers.buffer,
        "get_unprocessed_buffer_ids",
        fake_get_unprocessed_buffer_ids,
    )
    monkeypatch.setattr(
        buffer_api.controllers.buffer_background,
        "flush_buffer_background_running",
        fake_flush_buffer_background_running,
    )

    request = SimpleNamespace(state=SimpleNamespace(memobase_project_id="project-1"))

    response = await buffer_api.flush_buffer(
        request,
        user_id="user-1",
        buffer_type="chat",
        wait_process=False,
        background_tasks=FakeBackgroundTasks(),
    )

    assert response.errno == 0
    assert len(calls) == 1
    assert calls[0][1:] == ("user-1", "project-1", "chat")
