"""Tests for the Redis-backed JobStore — URL loaded from config/env."""
import pytest
from sleep_bci.api.job_store import JobStore
from sleep_bci.config import settings


@pytest.fixture(scope="module")
def store():
    s = JobStore(redis_url=settings.redis_url)
    if s.backend != "redis":
        pytest.skip(f"Redis not reachable at {settings.redis_url!r} — check REDIS_URL in .env")
    s.clear()
    yield s
    s.clear()


def test_backend_is_redis(store):
    assert store.backend == "redis"


def test_set_and_get(store):
    store.set("test-job-1", {"status": "queued", "progress": 0})
    result = store.get("test-job-1")
    assert result is not None
    assert result["status"] == "queued"
    assert result["progress"] == 0


def test_exists(store):
    store.set("test-job-2", {"status": "running"})
    assert store.exists("test-job-2") is True
    assert store.exists("does-not-exist") is False


def test_get_missing_returns_none(store):
    assert store.get("ghost-key-xyz") is None


def test_overwrite(store):
    store.set("test-job-3", {"status": "queued"})
    store.set("test-job-3", {"status": "succeeded"})
    assert store.get("test-job-3")["status"] == "succeeded"


def test_clear(store):
    store.set("test-job-4", {"status": "done"})
    store.clear()
    assert store.get("test-job-4") is None
