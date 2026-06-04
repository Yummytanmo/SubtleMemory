"""Subprocess bridge for native MIRIX ingestion and manager search."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class MirixNativeBackend:
    """Run native MIRIX in a per-conversation subprocess environment."""

    def __init__(
        self,
        *,
        mirix_root: Path,
        runtime_path: Path,
        agent_state_path: Path,
        config_path: Path,
        openai_api_key: str = "",
        openai_base_url: str = "",
        embedding_api_key: str = "",
        embedding_base_url: str = "",
        python_executable: Optional[Path] = None,
    ) -> None:
        self.mirix_root = Path(mirix_root).resolve()
        self.runtime_path = Path(runtime_path).resolve()
        self.agent_state_path = Path(agent_state_path).resolve()
        self.config_path = Path(config_path).resolve()
        self.openai_api_key = str(openai_api_key or "")
        self.openai_base_url = str(openai_base_url or "")
        self.embedding_api_key = str(embedding_api_key or "")
        self.embedding_base_url = str(embedding_base_url or "")
        self.python_executable = Path(python_executable).resolve() if python_executable else self._default_python()

    def ingest_conversation(
        self, *, conversation_id: str, sessions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        payload = {
            "conversation_id": conversation_id,
            "sessions": self._jsonable_sessions(sessions),
        }
        return self._run_subprocess("ingest", payload)

    def search(self, *, query: str, top_k: int) -> Dict[str, Any]:
        payload = {"query": query, "top_k": int(top_k)}
        return self._run_subprocess("search", payload)

    def answer_question(
        self, *, question: str, readback_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        payload = {"question": str(question or "")}
        if readback_context:
            payload["readback_context"] = readback_context
        return self._run_subprocess("answer", payload)

    def _run_subprocess(self, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.runtime_path.mkdir(parents=True, exist_ok=True)
        env = self._subprocess_env(operation=operation, payload=payload)
        command = [
            str(self.python_executable),
            str(Path(__file__).resolve()),
            "--operation",
            operation,
            "--mirix-root",
            str(self.mirix_root),
            "--runtime-path",
            str(self.runtime_path),
            "--agent-state-path",
            str(self.agent_state_path),
            "--config-path",
            str(self.config_path),
        ]
        proc = subprocess.run(
            command,
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            env=env,
            cwd=str(self._subprocess_cwd()),
            check=False,
        )
        if proc.returncode != 0:
            error_parts = []
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            if stderr:
                error_parts.append(f"stderr: {stderr[-4000:]}")
            if stdout:
                error_parts.append(f"stdout: {stdout[-4000:]}")
            detail = "\n".join(error_parts) or "<no subprocess output>"
            raise RuntimeError(
                f"MIRIX native {operation} failed with exit {proc.returncode}: "
                f"{detail}"
            )
        try:
            result, native_stdout = self._parse_subprocess_json(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"MIRIX native {operation} returned non-JSON output: "
                f"{proc.stdout[:1000]}"
            ) from exc
        if not isinstance(result, dict):
            raise RuntimeError(f"MIRIX native {operation} returned {type(result).__name__}")
        if result.get("status") == "error":
            raise RuntimeError(
                f"MIRIX native {operation} error: {result.get('error') or result}"
            )
        native_output: Dict[str, str] = {}
        if native_stdout:
            native_output["stdout"] = native_stdout[-4000:]
        if proc.stderr:
            native_output["stderr"] = proc.stderr[-4000:]
        if native_output:
            result["native_output"] = native_output
        return result

    @staticmethod
    def _parse_subprocess_json(stdout: str) -> tuple[Dict[str, Any], str]:
        text = str(stdout or "").strip()
        if not text:
            raise json.JSONDecodeError("empty stdout", stdout, 0)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for index in range(len(lines) - 1, -1, -1):
            line = lines[index]
            if not line.startswith("{"):
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                return parsed, "\n".join(lines[:index])
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError("stdout JSON is not an object", stdout, 0)
        return parsed, ""

    def _subprocess_env(
        self, *, operation: str = "", payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        env = dict(os.environ)
        home = self.runtime_path / "home"
        if operation == "answer":
            question_hash = hashlib.sha1(
                str((payload or {}).get("question") or "").encode("utf-8")
            ).hexdigest()[:16]
            home = self.runtime_path / "answer_homes" / question_hash
        mirix_home = home / ".mirix"
        mirix_home.mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(home)
        env["MIRIX_DIR"] = str(mirix_home)
        env["MEMGPT_CONFIG_PATH"] = str(mirix_home / "config")
        env.setdefault("PYTHONUSERBASE", str(Path.home() / ".local"))

        existing_pythonpath = env.get("PYTHONPATH", "")
        path_parts = [str(self.mirix_root)]
        if existing_pythonpath:
            path_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(path_parts)

        if self.openai_api_key:
            env["OPENAI_API_KEY"] = self.openai_api_key
        if self.openai_base_url:
            env["MIRIX_OPENAI_BASE_URL"] = self.openai_base_url
            env.setdefault("OPENAI_API_BASE", self.openai_base_url)
        if self.embedding_api_key:
            env["MIRIX_EMBEDDING_API_KEY"] = self.embedding_api_key
        if self.embedding_base_url:
            env["MIRIX_EMBEDDING_BASE_URL"] = self.embedding_base_url
        return env

    def _default_python(self) -> Path:
        mirix_venv_python = self.mirix_root / ".venv" / "bin" / "python"
        if mirix_venv_python.exists():
            return mirix_venv_python
        return Path(sys.executable)

    def _subprocess_cwd(self) -> Path:
        public_evaluations = self.mirix_root / "public_evaluations"
        if public_evaluations.is_dir():
            return public_evaluations
        return self.mirix_root

    @staticmethod
    def _jsonable_sessions(sessions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        jsonable = []
        for session in sessions:
            jsonable.append(
                {
                    key: value
                    for key, value in session.items()
                    if key not in {"messages"}
                }
            )
        return jsonable


def _native_ingest(
    *,
    mirix_root: Path,
    runtime_path: Path,
    agent_state_path: Path,
    config_path: Path,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    public_eval_path = mirix_root / "public_evaluations"
    if str(public_eval_path) not in sys.path:
        sys.path.insert(0, str(public_eval_path))
    if str(mirix_root) not in sys.path:
        sys.path.insert(0, str(mirix_root))

    from agent import AgentWrapper  # type: ignore

    if agent_state_path.exists():
        shutil.rmtree(agent_state_path)
    agent_state_path.mkdir(parents=True, exist_ok=True)

    agent = AgentWrapper("mirix", config_path=str(config_path))
    sessions = list(payload.get("sessions") or [])
    result_sessions: List[Dict[str, Any]] = []
    for session in sessions:
        source_metadata = dict(session.get("source_metadata") or {})
        response = agent.send_message(
            session.get("text") or "",
            memorizing=True,
            source_metadata=source_metadata,
        )
        result_session = dict(session)
        result_session["provider_response"] = _safe_json_value(response)
        result_sessions.append(result_session)

    agent.save_agent(str(agent_state_path))
    sqlite_path = agent_state_path / "sqlite.db"
    memory_refs = _collect_memory_refs(sqlite_path)
    for session in result_sessions:
        source_session_id = str(session.get("source_session_id") or "")
        session["memory_refs"] = [
            ref for ref in memory_refs if ref.get("source_session_id") == source_session_id
        ]

    return {
        "status": "ok",
        "runtime_path": str(runtime_path),
        "agent_state_path": str(agent_state_path),
        "sqlite_path": str(sqlite_path),
        "sessions": result_sessions,
    }


def _native_search(
    *,
    mirix_root: Path,
    agent_state_path: Path,
    config_path: Path,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    public_eval_path = mirix_root / "public_evaluations"
    if str(public_eval_path) not in sys.path:
        sys.path.insert(0, str(public_eval_path))
    if str(mirix_root) not in sys.path:
        sys.path.insert(0, str(mirix_root))

    from agent import AgentWrapper  # type: ignore

    query = str(payload.get("query") or "")
    top_k = int(payload.get("top_k") or 20)
    agent = AgentWrapper(
        "mirix",
        load_agent_from=str(agent_state_path),
        config_path=str(config_path),
    )
    native_agent = agent.agent
    server = native_agent.client.server
    states = native_agent.agent_states
    timezone_str = native_agent.client.server.user_manager.get_user_by_id(
        native_agent.client.user_id
    ).timezone

    results: List[Dict[str, Any]] = []
    manager_calls = (
        (
            "episodic",
            server.episodic_memory_manager.list_episodic_memory,
            states.episodic_memory_agent_state,
            "details",
        ),
        (
            "semantic",
            server.semantic_memory_manager.list_semantic_items,
            states.semantic_memory_agent_state,
            "details",
        ),
        (
            "resource",
            server.resource_memory_manager.list_resources,
            states.resource_memory_agent_state,
            "summary",
        ),
        (
            "procedural",
            server.procedural_memory_manager.list_procedures,
            states.procedural_memory_agent_state,
            "summary",
        ),
    )
    for memory_type, manager_fn, agent_state, search_field in manager_calls:
        memories = manager_fn(
            agent_state=agent_state,
            query=query,
            embedded_text=None,
            search_field=search_field,
            search_method="embedding",
            limit=top_k,
            timezone_str=timezone_str,
        )
        for memory in memories or []:
            results.append(_memory_to_result(memory_type, memory))

    return {
        "status": "ok",
        "agent_state_path": str(agent_state_path),
        "sqlite_path": str(agent_state_path / "sqlite.db"),
        "results": results,
    }


def _native_answer(
    *,
    mirix_root: Path,
    runtime_path: Path,
    agent_state_path: Path,
    config_path: Path,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    public_eval_path = mirix_root / "public_evaluations"
    if str(public_eval_path) not in sys.path:
        sys.path.insert(0, str(public_eval_path))
    if str(mirix_root) not in sys.path:
        sys.path.insert(0, str(mirix_root))

    from agent import AgentWrapper  # type: ignore

    question = str(payload.get("question") or "")
    readback_context = payload.get("readback_context")
    agent = AgentWrapper(
        "mirix",
        load_agent_from=str(agent_state_path),
        config_path=str(config_path),
    )
    try:
        agent.prepare_before_asking_questions()
    except AttributeError as exc:
        raise RuntimeError(
            "MIRIX public_evaluations AgentWrapper does not expose "
            "prepare_before_asking_questions(); cannot run native-agent answer "
            f"for agent_state_path={agent_state_path}"
        ) from exc
    if isinstance(readback_context, dict) and readback_context:
        answer = _native_answer_with_readback_context(agent, question, readback_context)
    else:
        answer = agent.send_message(question, memorizing=False)

    return {
        "status": "ok",
        "native_agent_answer": str(answer or ""),
        "runtime_path": str(runtime_path),
        "agent_state_path": str(agent_state_path),
        "sqlite_path": str(agent_state_path / "sqlite.db"),
    }


def _native_answer_with_readback_context(
    agent: Any,
    question: str,
    readback_context: Dict[str, Any],
) -> Any:
    formatted_context = str(readback_context.get("formatted_context") or "")
    results = readback_context.get("results") or []
    retrieved_memories = _build_native_readback_memories(
        formatted_context=formatted_context,
        results=results if isinstance(results, list) else [],
    )
    previous_include_recent_screenshots = getattr(agent, "include_recent_screenshots", None)
    if hasattr(agent, "include_recent_screenshots"):
        agent.include_recent_screenshots = False
    try:
        response = _answer_native_question_with_readback_memories_once(
            agent=agent,
            question=question,
            retrieved_memories=retrieved_memories,
        )
    finally:
        if previous_include_recent_screenshots is not None:
            agent.include_recent_screenshots = previous_include_recent_screenshots
    return response


def _answer_native_question_with_readback_memories_once(
    *, agent: Any, question: str, retrieved_memories: Dict[str, Any]
) -> Any:
    native_agent = getattr(agent, "agent", None)
    client = getattr(native_agent, "client", None)
    server = getattr(client, "server", None)
    agent_states = getattr(native_agent, "agent_states", None)
    chat_agent_state = getattr(agent_states, "agent_state", None)
    chat_agent_id = getattr(chat_agent_state, "id", None)
    if server is None or chat_agent_id is None:
        raise RuntimeError(
            "MIRIX readback one-shot answer requires access to the native "
            "chat agent server and chat_agent state"
        )

    chat_agent = server.load_agent(
        agent_id=chat_agent_id,
        interface=getattr(client, "interface", None),
        actor=getattr(client, "user", None),
    )
    in_context_messages = chat_agent.agent_manager.get_in_context_messages(
        agent_id=chat_agent.agent_state.id,
        actor=chat_agent.user,
    )
    if not in_context_messages:
        raise RuntimeError("MIRIX readback one-shot answer found no system prompt")

    raw_system = in_context_messages[0].content[0].text
    complete_system_prompt, _ = chat_agent.build_system_prompt_with_memories(
        raw_system=raw_system,
        retrieved_memories=retrieved_memories,
    )

    from mirix.helpers.message_helpers import prepare_input_message_create
    from mirix.schemas.enums import MessageRole
    from mirix.schemas.message import MessageCreate
    from mirix.schemas.mirix_message_content import TextContent

    system_message = copy.deepcopy(in_context_messages[0])
    system_message.content[0].text = complete_system_prompt
    user_message = prepare_input_message_create(
        MessageCreate(
            role=MessageRole.user,
            content=[TextContent(text=str(question or ""))],
        ),
        chat_agent.agent_state.id,
        wrap_user_message=False,
        wrap_system_message=True,
    )

    original_tools = list(getattr(chat_agent.agent_state, "tools", []) or [])
    try:
        chat_agent.agent_state.tools = []
        response = chat_agent._get_ai_reply(
            message_sequence=[system_message, user_message],
            first_message=False,
            stream=False,
        )
    finally:
        chat_agent.agent_state.tools = original_tools

    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    return str(getattr(message, "content", "") or "")


def _build_native_readback_memories(
    *,
    formatted_context: str,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    by_type: Dict[str, List[str]] = {
        "episodic": [],
        "semantic": [],
        "resource": [],
        "procedural": [],
    }
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        memory_type = str(
            metadata.get("memory_type") or metadata.get("kind") or "episodic"
        ).strip()
        if memory_type not in by_type:
            memory_type = "episodic"
        by_type[memory_type].append(f"[{index}] {content}")

    fallback = formatted_context.strip()
    if fallback and not any(by_type.values()):
        by_type["episodic"].append(fallback)

    episodic_context = "\n".join(by_type["episodic"]).strip()
    return {
        "key_words": "",
        "core": "",
        "knowledge_vault": "",
        "episodic": [episodic_context, episodic_context],
        "semantic": "\n".join(by_type["semantic"]).strip(),
        "resource": "\n".join(by_type["resource"]).strip(),
        "procedural": "\n".join(by_type["procedural"]).strip(),
    }


def _extract_native_response_text(response: Any) -> str:
    messages = getattr(response, "messages", None) or []
    for message in reversed(messages):
        message_type = str(getattr(message, "message_type", ""))
        if message_type.endswith("assistant_message"):
            return str(getattr(message, "content", "") or "")
    if messages:
        last = messages[-1]
        content = getattr(last, "content", None)
        if content is not None:
            return str(content)
    return str(response or "")


def _collect_memory_refs(sqlite_path: Path) -> List[Dict[str, Any]]:
    if not sqlite_path.exists():
        return []
    import sqlite3

    refs: List[Dict[str, Any]] = []
    table_specs = (
        ("episodic", "episodic_memory"),
        ("semantic", "semantic_memory"),
        ("resource", "resource_memory"),
        ("procedural", "procedural_memory"),
    )
    with sqlite3.connect(str(sqlite_path)) as conn:
        conn.row_factory = sqlite3.Row
        for memory_type, table in table_specs:
            try:
                rows = conn.execute(f"SELECT id, metadata_ FROM {table}").fetchall()
            except sqlite3.OperationalError:
                continue
            for row in rows:
                metadata = _json_load_dict(row["metadata_"])
                source_session_id = _source_session_id_from_metadata(metadata)
                refs.append(
                    {
                        "provider": "mirix",
                        "memory_id": str(row["id"]),
                        "memory_type": memory_type,
                        "source_session_id": source_session_id,
                        "sqlite_path": str(sqlite_path),
                        "agent_state_path": str(sqlite_path.parent),
                        "runtime_path": str(sqlite_path.parent.parent),
                    }
                )
    return refs


def _memory_to_result(memory_type: str, memory: Any) -> Dict[str, Any]:
    metadata = getattr(memory, "metadata_", None) or {}
    content = _memory_content(memory_type, memory)
    return {
        "memory_type": memory_type,
        "memory_id": str(getattr(memory, "id", "")),
        "content": content,
        "score": 1.0,
        "tree_path": list(getattr(memory, "tree_path", None) or []),
        "metadata": _safe_json_value(metadata),
    }


def _memory_content(memory_type: str, memory: Any) -> str:
    if memory_type in {"episodic", "semantic"}:
        return str(getattr(memory, "details", None) or getattr(memory, "summary", "") or "")
    if memory_type == "resource":
        return str(getattr(memory, "summary", None) or getattr(memory, "content", "") or "")
    if memory_type == "procedural":
        steps = getattr(memory, "steps", None)
        if isinstance(steps, list):
            return " ".join(str(step).strip() for step in steps if str(step).strip())
        return str(steps or getattr(memory, "summary", "") or "")
    return str(getattr(memory, "summary", "") or "")


def _safe_json_value(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _json_load_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value in (None, ""):
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _source_session_id_from_metadata(metadata: Dict[str, Any]) -> str:
    direct = str(metadata.get("source_session_id") or "").strip()
    if direct:
        return direct
    history = metadata.get("source_metadata_history") or []
    if isinstance(history, dict):
        history = [history]
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict):
                session_id = str(item.get("source_session_id") or "").strip()
                if session_id:
                    return session_id
    batch = metadata.get("source_metadata_batch") or []
    if isinstance(batch, list):
        for item in batch:
            if isinstance(item, dict):
                session_id = str(item.get("source_session_id") or "").strip()
                if session_id:
                    return session_id
    return ""


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["ingest", "search", "answer"], required=True)
    parser.add_argument("--mirix-root", required=True)
    parser.add_argument("--runtime-path", required=True)
    parser.add_argument("--agent-state-path", required=True)
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    payload = json.loads(sys.stdin.read() or "{}")
    try:
        if args.operation == "ingest":
            result = _native_ingest(
                mirix_root=Path(args.mirix_root),
                runtime_path=Path(args.runtime_path),
                agent_state_path=Path(args.agent_state_path),
                config_path=Path(args.config_path),
                payload=payload,
            )
        elif args.operation == "search":
            result = _native_search(
                mirix_root=Path(args.mirix_root),
                agent_state_path=Path(args.agent_state_path),
                config_path=Path(args.config_path),
                payload=payload,
            )
        else:
            result = _native_answer(
                mirix_root=Path(args.mirix_root),
                runtime_path=Path(args.runtime_path),
                agent_state_path=Path(args.agent_state_path),
                config_path=Path(args.config_path),
                payload=payload,
            )
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 1

    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
