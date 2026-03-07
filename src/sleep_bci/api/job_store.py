"""Redis-backed job store with in-memory fallback.

When REDIS_URL is set, jobs survive server restarts and container recycling.
When Redis is unavailable or not configured, an in-memory dict is used instead
(suitable for local development without a running Redis instance).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

TTL_SECONDS = 86400  # 24 hours


def _serialize(obj: Any) -> Any:
    """Custom JSON serializer for types not handled by the default encoder."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):  # Pydantic model
        return obj.model_dump()
    if hasattr(obj, "value"):       # Enum
        return obj.value
    raise TypeError(f"Cannot serialize type {type(obj)!r}")


class JobStore:
    """Stores job state in Redis (if available) or falls back to an in-memory dict.

    Usage:
        store = JobStore(redis_url="redis://localhost:6379")  # Redis
        store = JobStore()                                     # in-memory
    """

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self._redis = None
        self._mem: dict[str, dict] = {}

        if redis_url:
            try:
                import redis as _redis
                client = _redis.from_url(redis_url, decode_responses=True)
                client.ping()
                self._redis = client
                logger.info("JobStore: connected to Redis at %s", redis_url)
            except Exception as exc:
                logger.warning(
                    "JobStore: Redis unavailable (%s) — falling back to in-memory store", exc
                )

    @property
    def backend(self) -> str:
        return "redis" if self._redis else "memory"

    def set(self, job_id: str, data: dict) -> None:
        """Serialize and store a job dict."""
        payload = json.dumps(data, default=_serialize)
        if self._redis:
            self._redis.setex(job_id, TTL_SECONDS, payload)
        else:
            # Store a clean deserialized copy so in-memory behaviour mirrors Redis
            self._mem[job_id] = json.loads(payload)

    def get(self, job_id: str) -> Optional[dict]:
        """Return the stored job dict, or None if not found."""
        if self._redis:
            raw = self._redis.get(job_id)
            return json.loads(raw) if raw else None
        return self._mem.get(job_id)

    def exists(self, job_id: str) -> bool:
        if self._redis:
            return bool(self._redis.exists(job_id))
        return job_id in self._mem

    def clear(self) -> None:
        """Wipe all jobs. Intended for use in tests only."""
        if self._redis:
            self._redis.flushdb()
        else:
            self._mem.clear()
