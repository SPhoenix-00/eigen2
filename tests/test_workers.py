"""
Unit tests for training/workers.py — worker functions and SharedMemoryManager.
"""

import pytest
from collections import OrderedDict
from training.workers import (
    _cache_agent,
    _get_cached_agent,
    _WORKER_CACHE_MAX_SIZE,
)


class TestAgentCaching:
    def setup_method(self):
        """Reset the global worker cache before each test."""
        import training.workers as w
        w._worker_agent_cache = OrderedDict()

    def test_cache_and_retrieve(self):
        _cache_agent("hash1", "agent1")
        assert _get_cached_agent("hash1") == "agent1"

    def test_cache_miss(self):
        assert _get_cached_agent("nonexistent") is None

    def test_lru_eviction(self):
        """When cache exceeds max size, oldest entries are evicted."""
        import training.workers as w
        # Fill beyond capacity
        for i in range(_WORKER_CACHE_MAX_SIZE + 5):
            _cache_agent(f"hash_{i}", f"agent_{i}")
        # Oldest should be evicted
        assert _get_cached_agent("hash_0") is None
        assert _get_cached_agent("hash_1") is None
        # Newest should still be there
        assert _get_cached_agent(f"hash_{_WORKER_CACHE_MAX_SIZE + 4}") is not None

    def test_lru_access_moves_to_end(self):
        """Accessing an item should protect it from eviction."""
        import training.workers as w
        _cache_agent("keep_me", "protected_agent")
        # Fill rest of cache
        for i in range(_WORKER_CACHE_MAX_SIZE - 1):
            _cache_agent(f"fill_{i}", f"filler_{i}")
        # Access the first entry to move it to end
        result = _get_cached_agent("keep_me")
        assert result == "protected_agent"
        # Add more to trigger eviction of oldest fillers
        _cache_agent("new_entry", "new_agent")
        # keep_me should survive (it was recently accessed)
        assert _get_cached_agent("keep_me") == "protected_agent"


class TestSharedMemoryManager:
    def test_local_mode_skips_creation(self):
        """In local mode, no shared memory should be created."""
        from training.workers import SharedMemoryManager
        from unittest.mock import MagicMock

        mock_loader = MagicMock()
        manager = SharedMemoryManager(mock_loader, local_mode=True)
        assert manager._shm_blocks == []
        assert manager._shm_metadata == {}
        manager.cleanup()  # Should not raise
