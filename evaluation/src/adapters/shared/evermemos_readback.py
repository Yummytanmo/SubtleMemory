"""Shared EverMemOS /memories/get readback helpers."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from evaluation.src.core.readback import (
    NormalizedStorageObject,
    StorageReadbackResult,
)

EverMemOSRequestJson = Callable[[str, str, Dict[str, Any]], Awaitable[Dict[str, Any]]]


def clean_id(value: Any) -> str:
    return str(value or "").strip()


def session_filter_value(session_ids: Iterable[Any] | None) -> Any:
    normalized: List[str] = []
    seen: set[str] = set()
    for value in session_ids or []:
        clean = clean_id(value)
        if clean and clean not in seen:
            normalized.append(clean)
            seen.add(clean)
    if not normalized:
        return None
    if len(normalized) == 1:
        return normalized[0]
    return {"in": normalized}


def normalize_episode_object(
    episode: Dict[str, Any], *, provider: str
) -> NormalizedStorageObject:
    session_id = clean_id(episode.get("session_id"))
    content_parts = [
        clean_id(episode.get("timestamp")),
        clean_id(
            episode.get("episode")
            or episode.get("summary")
            or episode.get("subject")
        ),
    ]
    content = ": ".join(part for part in content_parts if part)
    return NormalizedStorageObject(
        session_id=session_id,
        kind="episodic_memory",
        id=clean_id(episode.get("id")),
        content=content,
        metadata={
            "provider": provider,
            "user_id": episode.get("user_id"),
            "group_id": episode.get("group_id"),
            "memory_type": episode.get("type"),
            "parent_type": episode.get("parent_type"),
            "parent_id": episode.get("parent_id"),
        },
        raw=episode,
    )


def user_ids_from_manifest(
    *,
    import_manifest_rows: Iterable[Dict[str, Any]],
    target_session_ids: Iterable[str],
) -> List[str]:
    target_set = {clean_id(value) for value in target_session_ids if clean_id(value)}
    user_ids: List[str] = []
    for row in import_manifest_rows or []:
        for ref in row.get("memory_refs", []) or []:
            if not isinstance(ref, dict):
                continue
            ref_user_id = clean_id(ref.get("user_id"))
            ref_session_id = clean_id(
                ref.get("session_id")
                or ref.get("source_session_id")
                or ref.get("run_id")
            )
            if target_set and ref_session_id not in target_set:
                continue
            if ref_user_id and ref_user_id not in user_ids:
                user_ids.append(ref_user_id)
    return user_ids


async def get_evermemos_storage_readback(
    *,
    request_json: EverMemOSRequestJson,
    provider: str,
    scope: str,
    user_id: Optional[str] = None,
    session_ids: Optional[List[str]] = None,
    question_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    page_size: int = 100,
    max_pages: int = 20,
) -> StorageReadbackResult:
    """Read EverMemOS episodic memories through /memories/get."""
    context = context or {}
    target_session_ids = [
        clean_id(value)
        for value in (session_ids or context.get("session_ids") or [])
        if clean_id(value)
    ]
    checked_session_ids = list(dict.fromkeys(target_session_ids))
    normalized_scope = clean_id(scope).lower() or "personal"
    if normalized_scope == "group":
        return StorageReadbackResult(
            status="unsupported",
            checked_session_ids=checked_session_ids,
            metadata={
                "provider": provider,
                "question_id": question_id,
                "reason": "EverMemOS group API does not provide reliable benchmark session readback.",
            },
        )

    user_ids: List[str] = []
    for candidate in (
        user_id,
        context.get("user_id"),
        *((context.get("user_ids") or []) if isinstance(context.get("user_ids"), list) else []),
    ):
        clean = clean_id(candidate)
        if clean and clean not in user_ids:
            user_ids.append(clean)
    for candidate in user_ids_from_manifest(
        import_manifest_rows=context.get("import_manifest_rows", []) or [],
        target_session_ids=checked_session_ids,
    ):
        if candidate not in user_ids:
            user_ids.append(candidate)

    if not user_ids:
        return StorageReadbackResult(
            status="no_user_id",
            checked_session_ids=checked_session_ids,
            missing_session_ids=checked_session_ids,
            metadata={"provider": provider, "question_id": question_id},
            errors=[
                {
                    "stage": "get_storage_readback",
                    "error_type": "missing_user_id",
                    "error_message": "Cannot determine EverMemOS personal user_id for readback.",
                }
            ],
        )

    page_size = max(1, min(int(page_size), 100))
    max_pages = max(1, int(max_pages))

    objects: List[NormalizedStorageObject] = []
    errors: List[Dict[str, Any]] = []
    readback_filters: List[Dict[str, Any]] = []

    for uid in user_ids:
        filters: Dict[str, Any] = {"user_id": uid}
        session_filter = session_filter_value(checked_session_ids)
        if session_filter:
            filters["session_id"] = session_filter
        readback_filters.append(dict(filters))

        page = 1
        while page <= max_pages:
            try:
                payload = await request_json(
                    "POST",
                    "/memories/get",
                    {
                        "memory_type": "episodic_memory",
                        "filters": filters,
                        "page": page,
                        "page_size": page_size,
                        "rank_by": "timestamp",
                        "rank_order": "desc",
                    },
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {
                        "stage": "get_storage_readback",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "filters": filters,
                    }
                )
                break

            data = payload.get("data") if isinstance(payload, dict) else {}
            data = data or {}
            episodes = data.get("episodes") or []
            for episode in episodes:
                if isinstance(episode, dict):
                    objects.append(
                        normalize_episode_object(episode, provider=provider)
                    )

            total_count = int(data.get("total_count") or 0)
            count = int(data.get("count") or len(episodes))
            if not episodes or count < page_size or page * page_size >= total_count:
                break
            page += 1

    object_session_ids = {
        obj.session_id for obj in objects if obj.session_id and obj.session_id != "-1"
    }
    missing_session_ids = [
        session_id for session_id in checked_session_ids if session_id not in object_session_ids
    ]

    status = "ok"
    if errors and not objects:
        status = "error"
    elif missing_session_ids:
        status = "missing_sessions"

    return StorageReadbackResult(
        status=status,
        checked_session_ids=checked_session_ids,
        missing_session_ids=missing_session_ids,
        objects=objects,
        metadata={
            "provider": provider,
            "question_id": question_id,
            "user_ids": user_ids,
            "readback_scope": "session" if checked_session_ids else "user",
            "filters": readback_filters,
            "object_count": len(objects),
        },
        errors=errors,
    )
