"""
Unit tests — VectorStore
=========================
Tests Qdrant integration with a mocked client.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.services.vector_store import VectorStore, _identity_to_point_id


class TestIdentityToPointId:
    def test_deterministic(self):
        """Same input must always produce same UUID."""
        uid1 = _identity_to_point_id("alice_001")
        uid2 = _identity_to_point_id("alice_001")
        assert uid1 == uid2

    def test_different_ids_produce_different_uuids(self):
        uid1 = _identity_to_point_id("alice_001")
        uid2 = _identity_to_point_id("bob_002")
        assert uid1 != uid2

    def test_valid_uuid_format(self):
        import uuid
        uid = _identity_to_point_id("test_identity")
        # Should not raise
        uuid.UUID(uid)


class TestVectorStoreSearch:
    def _make_store_with_mock_client(self):
        store = VectorStore.__new__(VectorStore)
        store._client = MagicMock()
        return store

    def test_search_returns_sorted_candidates(self):
        store = self._make_store_with_mock_client()

        hit1 = MagicMock()
        hit1.score = 0.75
        hit1.payload = {"identity_id": "alice", "metadata": {}, "enrolled_at": "2024-01-01"}

        hit2 = MagicMock()
        hit2.score = 0.45
        hit2.payload = {"identity_id": "bob", "metadata": {}, "enrolled_at": "2024-01-01"}

        store._client.search.return_value = [hit1, hit2]

        embedding = np.random.randn(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        results, latency_ms = store.search(embedding, top_k=2)

        assert len(results) == 2
        assert results[0]["identity_id"] == "alice"
        assert results[0]["cosine_score"] == pytest.approx(0.75, abs=1e-4)
        assert results[1]["identity_id"] == "bob"
        assert latency_ms > 0

    def test_upsert_calls_qdrant_upsert(self):
        store = self._make_store_with_mock_client()
        store._client.retrieve.return_value = []  # doesn't exist yet

        embedding = np.random.randn(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        already_existed = store.upsert("carol_003", embedding, {"dept": "eng"})

        assert already_existed is False
        store._client.upsert.assert_called_once()

    def test_upsert_detects_existing_identity(self):
        store = self._make_store_with_mock_client()
        store._client.retrieve.return_value = [MagicMock()]  # already exists

        embedding = np.random.randn(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        already_existed = store.upsert("dave_004", embedding)
        assert already_existed is True

    def test_count_returns_integer(self):
        store = self._make_store_with_mock_client()
        store._client.count.return_value = MagicMock(count=42)
        assert store.count() == 42

    def test_delete_existing_identity(self):
        store = self._make_store_with_mock_client()
        store._client.retrieve.return_value = [MagicMock()]

        result = store.delete("alice_001")

        assert result is True
        store._client.delete.assert_called_once()

    def test_delete_nonexistent_identity(self):
        store = self._make_store_with_mock_client()
        store._client.retrieve.return_value = []

        result = store.delete("ghost_999")
        assert result is False
        store._client.delete.assert_not_called()
