import uuid
import asyncio
import traceback
from ..env import BufferStatus, TRACE_LOG
from ..models.database import BufferZone
from ..models.blob import BlobType
from ..connectors import Session, PROJECT_ID, get_redis_client
from .modal import BLOBS_PROCESS
from .buffer import flush_buffer_by_ids

REDIS_LUA_CHECK_AND_DELETE_LOCK = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


def get_user_lock_key(user_id: str, project_id: str, scope: str) -> str:
    return f"memobase:user_lock:{PROJECT_ID}:{scope}:{project_id}:{user_id}"


def get_user_buffer_queue_key(user_id: str, project_id: str, scope: str) -> str:
    return f"memobase:user_buffer_queue:{PROJECT_ID}:{scope}:{project_id}:{user_id}"


def pack_ids_to_str(ids: list[str]) -> str:
    return "::".join([str(i) for i in ids])


def unpack_ids_from_str(ids_str: str) -> list[str]:
    return [i.strip() for i in ids_str.split("::") if i.strip()]


async def flush_buffer_by_ids_in_background(
    user_id: str, project_id: str, blob_type: BlobType, buffer_ids: list[str]
) -> None:
    if not len(buffer_ids):
        return
    if blob_type not in BLOBS_PROCESS:
        return

    # 1. mark buffer as processing
    with Session() as session:
        buffer_blob_data = (
            session.query(BufferZone.id)
            .filter(
                BufferZone.user_id == user_id,
                BufferZone.blob_type == str(blob_type),
                BufferZone.project_id == project_id,
                BufferZone.status == BufferStatus.idle,
                BufferZone.id.in_(buffer_ids),
            )
            .order_by(BufferZone.created_at)
            .all()
        )
        actual_buffer_ids = [row.id for row in buffer_blob_data]
        if not len(actual_buffer_ids):
            return
        session.query(BufferZone).filter(
            BufferZone.id.in_(actual_buffer_ids),
        ).update(
            {BufferZone.status: BufferStatus.processing},
            synchronize_session=False,
        )

        session.commit()

    # 2. add actual buffer ids to a redis queue
    buffer_queue_key = get_user_buffer_queue_key(
        user_id, project_id, f"flush_buffer_background_{blob_type}"
    )
    buffer_id_batches = _chunk_ids(actual_buffer_ids)

    try:
        async with get_redis_client() as redis_client:
            for buffer_ids in buffer_id_batches:
                await redis_client.rpush(buffer_queue_key, pack_ids_to_str(buffer_ids))
            queue_size = await redis_client.llen(buffer_queue_key)

            TRACE_LOG.info(
                project_id,
                user_id,
                f"[background] Enqueued {len(actual_buffer_ids)} buffer IDs in {len(buffer_id_batches)} batches to queue (queue size: {queue_size})",
            )

        await flush_buffer_background_running(user_id, project_id, blob_type)
    except Exception as e:
        TRACE_LOG.error(
            project_id,
            user_id,
            f"[background] Error enqueue buffer ids: {e}: {traceback.format_exc()}",
        )


def _chunk_ids(ids: list[str], batch_size: int = 8) -> list[list[str]]:
    return [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]


def _get_buffer_ids_by_status(
    user_id: str,
    project_id: str,
    blob_type: BlobType,
    status: str,
) -> list[str]:
    with Session() as session:
        rows = (
            session.query(BufferZone.id)
            .filter(
                BufferZone.user_id == user_id,
                BufferZone.blob_type == str(blob_type),
                BufferZone.project_id == project_id,
                BufferZone.status == status,
            )
            .order_by(BufferZone.created_at)
            .all()
        )
    return [str(row.id) for row in rows]


async def _enqueue_processing_buffers_if_queue_empty(
    redis_client,
    buffer_queue_key: str,
    user_id: str,
    project_id: str,
    blob_type: BlobType,
    batch_size: int = 8,
) -> int:
    if await redis_client.llen(buffer_queue_key):
        return 0

    processing_ids = _get_buffer_ids_by_status(
        user_id, project_id, blob_type, BufferStatus.processing
    )
    if not processing_ids:
        return 0

    for buffer_ids in _chunk_ids(processing_ids, batch_size=batch_size):
        await redis_client.rpush(buffer_queue_key, pack_ids_to_str(buffer_ids))
    return len(processing_ids)


async def flush_buffer_background_running(
    user_id: str,
    project_id: str,
    blob_type: BlobType,
    asleep_waiting_s: float = 0.1,  # Increased from 0.001 to reduce CPU usage
    max_iterations: int = 200,  # Maximum 200 tasks for this run, return after it reaches.
    process_interval_s: float = 60 * 5,  # Reduced from 10 minutes to 5 minutes
    max_processing_time_s: float = 60 * 15,  # Maximum 15 minutes total processing time
    max_consecutive_errors=5,  # Stop after 5 consecutive errors
):
    user_key = get_user_lock_key(
        user_id, project_id, f"flush_buffer_background_{blob_type}"
    )
    buffer_queue_key = get_user_buffer_queue_key(
        user_id, project_id, f"flush_buffer_background_{blob_type}"
    )

    __lock_value = str(uuid.uuid4())
    start_time = asyncio.get_event_loop().time()

    # Shorter lock timeout with renewal
    async with get_redis_client() as redis_client:
        acquired = await redis_client.set(
            user_key, __lock_value, nx=True, ex=process_interval_s
        )
        if not acquired:
            TRACE_LOG.debug(
                project_id,
                user_id,
                f"[background] Lock already acquired",
            )
            return

    try:
        iteration_count = 0
        consecutive_errors = 0

        while iteration_count < max_iterations:
            current_time = asyncio.get_event_loop().time()

            # Check lock and get next batch
            async with get_redis_client() as redis_client:
                recovered_count = await _enqueue_processing_buffers_if_queue_empty(
                    redis_client,
                    buffer_queue_key,
                    user_id,
                    project_id,
                    blob_type,
                )
                if recovered_count:
                    TRACE_LOG.warning(
                        project_id,
                        user_id,
                        f"[background] Re-enqueued {recovered_count} processing buffer IDs from an empty queue",
                    )

                lock_value = await redis_client.get(user_key)
                if lock_value is None or lock_value != __lock_value:  # Lock is expired
                    TRACE_LOG.debug(
                        project_id,
                        user_id,
                        "[background] Lock expired",
                    )
                    break

                buffer_ids_str = await redis_client.lpop(buffer_queue_key)
                if buffer_ids_str is None:  # Queue is empty
                    TRACE_LOG.debug(
                        project_id,
                        user_id,
                        "[background] Queue empty",
                    )
                    break

                # Renew lock timeout if needed
                await redis_client.expire(user_key, process_interval_s)
                current_queue_size = await redis_client.llen(buffer_queue_key)

            TRACE_LOG.info(
                project_id,
                user_id,
                f"[background]({iteration_count}/{max_iterations}) Processing buffer (left queue size: {current_queue_size})",
            )

            buffer_ids = unpack_ids_from_str(buffer_ids_str or "")
            if not buffer_ids:
                continue

            try:
                # Process the buffer with timeout protection
                processing_start = asyncio.get_event_loop().time()

                async def renew_lock_until_done():
                    while True:
                        try:
                            async with get_redis_client() as redis_client:
                                lock_value = await redis_client.get(user_key)
                                if lock_value is None or lock_value != __lock_value:
                                    return
                                await redis_client.expire(user_key, process_interval_s)
                            await asyncio.sleep(max(1.0, process_interval_s / 3))
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            TRACE_LOG.error(
                                project_id,
                                user_id,
                                f"[background] Failed to renew lock while processing: {e}",
                            )
                            await asyncio.sleep(1)

                renew_task = asyncio.create_task(renew_lock_until_done())
                p = await flush_buffer_by_ids(
                    user_id,
                    project_id,
                    blob_type,
                    buffer_ids,
                    select_status=BufferStatus.processing,
                )
                renew_task.cancel()
                try:
                    await renew_task
                except asyncio.CancelledError:
                    pass

                processing_time = asyncio.get_event_loop().time() - processing_start

                if not p.ok():
                    consecutive_errors += 1
                    TRACE_LOG.error(
                        project_id,
                        user_id,
                        f"[background] Error flushing buffer by ids: {p.msg()}",
                    )

                    # Stop if too many consecutive errors
                    if consecutive_errors >= max_consecutive_errors:
                        TRACE_LOG.error(
                            project_id,
                            user_id,
                            f"[background] Too many consecutive errors ({consecutive_errors}), stopping",
                        )
                        break
                else:
                    consecutive_errors = 0  # Reset error counter on success
                    TRACE_LOG.debug(
                        project_id,
                        user_id,
                        f"[background] Processed batch in {processing_time:.2f}s",
                    )

            except Exception as e:
                if "renew_task" in locals() and not renew_task.done():
                    renew_task.cancel()
                consecutive_errors += 1
                TRACE_LOG.error(
                    project_id,
                    user_id,
                    f"[background] Unknown Error flushing buffer by ids: {e}\n{traceback.format_exc()}",
                )

                # Stop if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    TRACE_LOG.error(
                        project_id,
                        user_id,
                        f"[background] Too many consecutive errors ({consecutive_errors}), stopping",
                    )
                    break

            # Sleep between iterations to prevent overwhelming the system
            await asyncio.sleep(asleep_waiting_s)
            iteration_count += 1

            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > max_processing_time_s:
                async with get_redis_client() as redis_client:
                    queue_size = await redis_client.llen(buffer_queue_key)
                if queue_size:
                    TRACE_LOG.warning(
                        project_id,
                        user_id,
                        f"[background] Maximum processing time ({max_processing_time_s}s) exceeded with {queue_size} queued batches left",
                    )
                    break

        total_processing_time = asyncio.get_event_loop().time() - start_time
        TRACE_LOG.info(
            project_id,
            user_id,
            f"[background] Completed processing. "
            f"Iterations: {iteration_count}, Time: {total_processing_time:.2f}s, "
            f"Final consecutive errors: {consecutive_errors}",
        )

    finally:
        try:
            async with get_redis_client() as redis_client:
                result = await redis_client.eval(
                    REDIS_LUA_CHECK_AND_DELETE_LOCK, 1, user_key, __lock_value
                )
                if result == 1:
                    TRACE_LOG.debug(
                        project_id,
                        user_id,
                        f"[background] Successfully released lock",
                    )
                else:
                    TRACE_LOG.warning(
                        project_id,
                        user_id,
                        f"[background] Lock was already expired/released",
                    )
        except Exception as e:
            TRACE_LOG.error(
                project_id,
                user_id,
                f"[background] Failed to release lock: {e}",
            )
