from pydantic import ValidationError
from typing import Optional
from ..models.database import UserEvent, UserEventGist
from ..models.response import (
    UserEventData,
    UserEventGistsData,
    UserEventGistData,
)
from ..models.utils import Promise, CODE
from ..connectors import Session
from ..utils import get_encoded_tokens, event_str_repr, event_embedding_str

from ..llms.embeddings import get_embedding
from datetime import timedelta
from sqlalchemy import desc, select
from sqlalchemy.sql import func
from ..env import TRACE_LOG, CONFIG


async def get_user_event_gists(
    user_id: str,
    project_id: str,
    topk: int = 10,
    time_range_in_days: int = 21,
) -> Promise[UserEventGistsData]:
    with Session() as session:
        query = (
            session.query(UserEventGist)
            .filter_by(user_id=user_id, project_id=project_id)
            .filter(
                UserEventGist.created_at
                > (func.now() - timedelta(days=time_range_in_days))
            )
        )
        user_event_gists = (
            query.order_by(UserEventGist.created_at.desc()).limit(topk).all()
        )
        if user_event_gists is None:
            return Promise.resolve(UserEventGistsData(gists=[]))
        results = [
            {
                "id": ue.id,
                "event_id": ue.event_id,
                "gist_data": ue.gist_data,
                "created_at": ue.created_at,
                "updated_at": ue.updated_at,
            }
            for ue in user_event_gists
        ]
    gists = UserEventGistsData(gists=results)
    return Promise.resolve(gists)


def _event_source_sessions(event_data: dict) -> set[str]:
    values = (event_data or {}).get("source_session_ids") or []
    if isinstance(values, str):
        values = [values]
    return {str(value) for value in values if str(value or "").strip()}


def _matches_any_session(event_data: dict, session_ids: set[str]) -> bool:
    if not session_ids:
        return True
    return bool(_event_source_sessions(event_data) & session_ids)


async def get_user_event_gists_by_sessions(
    user_id: str,
    project_id: str,
    session_ids: list[str],
    topk: Optional[int] = None,
    time_range_in_days: int = 21,
) -> Promise[UserEventGistsData]:
    target_session_ids = {str(value).strip() for value in session_ids if str(value).strip()}
    with Session() as session:
        rows = (
            session.query(UserEventGist, UserEvent)
            .join(
                UserEvent,
                (UserEventGist.event_id == UserEvent.id)
                & (UserEventGist.project_id == UserEvent.project_id),
            )
            .filter(
                UserEventGist.user_id == user_id,
                UserEventGist.project_id == project_id,
                UserEventGist.created_at
                > (func.now() - timedelta(days=time_range_in_days)),
            )
            .order_by(UserEventGist.created_at.desc())
            .all()
        )

        gists = []
        events_by_id = {}
        for gist, event in rows:
            if not _matches_any_session(event.event_data, target_session_ids):
                continue
            gists.append(
                UserEventGistData(
                    id=gist.id,
                    event_id=gist.event_id,
                    gist_data=gist.gist_data,
                    created_at=gist.created_at,
                    updated_at=gist.updated_at,
                )
            )
            events_by_id[str(event.id)] = UserEventData(
                id=event.id,
                event_data=event.event_data,
                created_at=event.created_at,
                updated_at=event.updated_at,
            )
            if topk is not None and len(gists) >= topk:
                break

    return Promise.resolve(
        UserEventGistsData(gists=gists, events=list(events_by_id.values()))
    )


async def truncate_event_gists(
    events: UserEventGistsData,
    max_token_size: int | None,
) -> Promise[UserEventGistsData]:
    if max_token_size is None:
        return Promise.resolve(events)
    c_tokens = 0
    truncated_results = []
    for r in events.gists:
        c_tokens += len(get_encoded_tokens(r.gist_data.content))
        if c_tokens > max_token_size:
            break
        truncated_results.append(r)
    events.gists = truncated_results
    return Promise.resolve(events)


async def search_user_event_gists(
    user_id: str,
    project_id: str,
    query: str,
    topk: int = 10,
    similarity_threshold: float = 0.2,
    time_range_in_days: int = 21,
) -> Promise[UserEventGistsData]:
    if not CONFIG.enable_event_embedding:
        TRACE_LOG.warning(
            project_id,
            user_id,
            "Event embedding is not enabled, skip search",
        )
        return Promise.reject(
            CODE.NOT_IMPLEMENTED,
            "Event embedding is not enabled",
        )
    query_embeddings = await get_embedding(
        project_id, [query], phase="query", model=CONFIG.embedding_model
    )
    if not query_embeddings.ok():
        TRACE_LOG.error(
            project_id,
            user_id,
            f"Failed to get embeddings: {query_embeddings.msg()}",
        )
        return query_embeddings
    query_embedding = query_embeddings.data()[0]

    # Calculate the time cutoff once
    time_cutoff = func.now() - timedelta(days=time_range_in_days)

    # Store the similarity expression to avoid recomputation
    similarity_expr = 1 - UserEventGist.embedding.cosine_distance(query_embedding)

    stmt = (
        select(
            UserEventGist,
            similarity_expr.label("similarity"),
        )
        .where(
            UserEventGist.user_id == user_id,
            UserEventGist.project_id == project_id,
            UserEventGist.created_at > time_cutoff,
            similarity_expr > similarity_threshold,
            UserEventGist.embedding.is_not(None),  # Skip null embeddings
        )
        .order_by(desc("similarity"))
        .limit(topk)
    )

    with Session() as session:
        # Use .all() instead of .scalars().all() to get both columns
        result = session.execute(stmt).all()
        user_event_gists: list[UserEventGistData] = []
        for row in result:
            user_event: UserEventGist = row[0]  # UserEventGist object
            similarity: float = row[1]  # similarity value
            user_event_gists.append(
                UserEventGistData(
                    id=user_event.id,
                    event_id=user_event.event_id,
                    gist_data=user_event.gist_data,
                    created_at=user_event.created_at,
                    updated_at=user_event.updated_at,
                    similarity=similarity,
                )
            )

        # Create UserEventsData with the events
        user_event_gists_data = UserEventGistsData(gists=user_event_gists)
        TRACE_LOG.info(
            project_id,
            user_id,
            f"Event Query: {query}",
        )

    return Promise.resolve(user_event_gists_data)
