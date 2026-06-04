import asyncio
from ...project import get_project_profile_config
from ....connectors import Session
from ....env import ProfileConfig, CONFIG, TRACE_LOG
from ....utils import get_blob_str, get_encoded_tokens
from ....models.blob import Blob
from ....models.utils import Promise, CODE
from ....models.response import IdsData, ChatModalResponse, UserProfilesData
from ...profile import add_update_delete_user_profiles
from ...event import append_user_event
from ...profile import get_user_profiles
from .extract import extract_topics

# from .merge import merge_or_valid_new_memos
from .merge_yolo import merge_or_valid_new_memos
from .summary import re_summary
from .organize import organize_profiles
from .types import MergeAddResult
from .event_summary import tag_event
from .entry_summary import entry_chat_summary


def _dedupe_nonempty(values: list | tuple) -> list[str]:
    result = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def extract_blob_provenance(blobs: list[Blob]) -> dict[str, list[str]]:
    source_session_ids = []
    source_blob_ids = []
    source_unit_ids = []
    for blob in blobs:
        fields = blob.fields or {}
        source_session_ids.append(
            fields.get("source_session_id") or fields.get("session_id")
        )
        source_blob_ids.append(fields.get("source_blob_id") or fields.get("blob_id"))
        unit_ids = fields.get("source_unit_ids") or fields.get("source_unit_id") or []
        if isinstance(unit_ids, list):
            source_unit_ids.extend(unit_ids)
        else:
            source_unit_ids.append(unit_ids)
    return {
        "source_session_ids": _dedupe_nonempty(source_session_ids),
        "source_blob_ids": _dedupe_nonempty(source_blob_ids),
        "source_unit_ids": _dedupe_nonempty(source_unit_ids),
    }


def truncate_chat_blobs(
    blobs: list[Blob], max_token_size: int
) -> tuple[list[str], list[Blob]]:
    results = []
    total_token_size = 0
    for b in blobs[::-1]:
        ts = len(get_encoded_tokens(get_blob_str(b)))
        total_token_size += ts
        if total_token_size <= max_token_size:
            results.append(b)
        else:
            break
    return results[::-1]


async def process_blobs(
    user_id: str, project_id: str, blobs: list[Blob]
) -> Promise[ChatModalResponse]:
    # 1. Extract patch profiles
    blobs = truncate_chat_blobs(blobs, CONFIG.max_chat_blob_buffer_process_token_size)
    if len(blobs) == 0:
        return Promise.reject(
            CODE.SERVER_PARSE_ERROR, "No blobs to process after truncating"
        )

    p = await get_project_profile_config(project_id)
    if not p.ok():
        return p
    project_profiles = p.data()

    p = await get_user_profiles(user_id, project_id)
    if not p.ok():
        return p
    current_user_profiles = p.data()

    p = await entry_chat_summary(
        user_id, project_id, blobs, project_profiles, current_user_profiles
    )
    if not p.ok():
        return p
    user_memo_str = p.data().strip()

    if not user_memo_str:
        return Promise.resolve(
            ChatModalResponse(
                event_id=None,
                add_profiles=[],
                update_profiles=[],
                delete_profiles=[],
            )
        )

    processing_results = await asyncio.gather(
        process_profile_res(
            user_id, project_id, user_memo_str, project_profiles, current_user_profiles
        ),
        process_event_res(
            user_id, project_id, user_memo_str, project_profiles, current_user_profiles
        ),
    )

    profile_results: Promise = processing_results[0]
    event_results: Promise = processing_results[1]

    if not profile_results.ok() or not event_results.ok():
        return Promise.reject(
            CODE.SERVER_PARSE_ERROR,
            f"Failed to process profile or event: {profile_results.msg()}, {event_results.msg()}",
        )

    intermediate_profile, delta_profile_data = profile_results.data()
    event_tags = event_results.data()

    p = await handle_session_event(
        user_id,
        project_id,
        user_memo_str,
        delta_profile_data,
        event_tags,
        project_profiles,
        extract_blob_provenance(blobs),
    )
    if not p.ok():
        return p
    eid = p.data()

    p = await handle_user_profile_db(user_id, project_id, intermediate_profile)
    if not p.ok():
        return p
    return Promise.resolve(
        ChatModalResponse(
            event_id=eid,
            add_profiles=p.data().ids,
            update_profiles=[up["profile_id"] for up in intermediate_profile["update"]],
            delete_profiles=intermediate_profile["delete"],
        )
    )


async def process_profile_res(
    user_id: str,
    project_id: str,
    user_memo_str: str,
    project_profiles: ProfileConfig,
    current_user_profiles: UserProfilesData,
) -> Promise[tuple[MergeAddResult, list[dict]]]:

    p = await extract_topics(
        user_id, project_id, user_memo_str, project_profiles, current_user_profiles
    )
    if not p.ok():
        return p
    extracted_data = p.data()

    # 2. Merge it to thw whole profile
    p = await merge_or_valid_new_memos(
        user_id,
        project_id,
        fact_contents=extracted_data["fact_contents"],
        fact_attributes=extracted_data["fact_attributes"],
        profiles=extracted_data["profiles"],
        config=project_profiles,
        total_profiles=extracted_data["total_profiles"],
    )
    if not p.ok():
        return p

    intermediate_profile = p.data()
    delta_profile_data = [
        p for p in (intermediate_profile["add"] + intermediate_profile["update_delta"])
    ]

    # 3. Check if we need to organize profiles
    p = await organize_profiles(
        user_id,
        project_id,
        intermediate_profile,
        config=project_profiles,
    )
    if not p.ok():
        TRACE_LOG.error(
            project_id,
            user_id,
            f"Failed to organize profiles: {p.msg()}",
        )

    # 4. Re-summary profiles if any slot is too big
    p = await re_summary(
        user_id,
        project_id,
        add_profile=intermediate_profile["add"],
        update_profile=intermediate_profile["update"],
    )
    if not p.ok():
        TRACE_LOG.error(
            project_id,
            user_id,
            f"Failed to re-summary profiles: {p.msg()}",
        )

    return Promise.resolve((intermediate_profile, delta_profile_data))


async def process_event_res(
    user_id: str,
    project_id: str,
    memo_str: str,
    config: ProfileConfig,
    current_user_profiles: UserProfilesData,
) -> Promise[list | None]:
    p = await tag_event(project_id, config, memo_str)
    if not p.ok():
        TRACE_LOG.error(
            project_id,
            user_id,
            f"Failed to tag event: {p.msg()}",
        )
        return p
    event_tags = p.data()
    return Promise.resolve(event_tags)


async def handle_session_event(
    user_id: str,
    project_id: str,
    memo_str: str,
    delta_profile_data: list[dict],
    event_tags: list | None,
    config: ProfileConfig,
    provenance: dict[str, list[str]] | None = None,
) -> Promise[str]:

    eid = await append_user_event(
        user_id,
        project_id,
        {
            "event_tip": memo_str,
            "event_tags": event_tags,
            "profile_delta": delta_profile_data,
            **(provenance or {}),
        },
    )

    return eid


async def handle_user_profile_db(
    user_id: str, project_id: str, intermediate_profile: MergeAddResult
) -> Promise[IdsData]:
    TRACE_LOG.info(
        project_id,
        user_id,
        f"Adding {len(intermediate_profile['add'])}, updating {len(intermediate_profile['update'])}, deleting {len(intermediate_profile['delete'])} profiles",
    )

    p = await add_update_delete_user_profiles(
        user_id,
        project_id,
        [ap["content"] for ap in intermediate_profile["add"]],
        [ap["attributes"] for ap in intermediate_profile["add"]],
        [up["profile_id"] for up in intermediate_profile["update"]],
        [up["content"] for up in intermediate_profile["update"]],
        [up["attributes"] for up in intermediate_profile["update"]],
        intermediate_profile["delete"],
    )
    return p
