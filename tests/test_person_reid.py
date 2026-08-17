"""
Unit Tests for Person ReID & Identity Intelligence Module
"""

import pytest
import numpy as np
from person_intelligence.reid_face_extractor import HybridPersonIDExtractor, IdentityCentroidTracker
from person_intelligence.vector_db_manager import VectorDBManager


def test_hybrid_feature_extraction():
    extractor = HybridPersonIDExtractor(face_dim=512, reid_dim=256)
    feats = extractor.extract_hybrid_features(identity_hint="TEST_SUBJECT_01")
    
    assert feats["face_embedding"].shape == (512,)
    assert feats["reid_embedding"].shape == (256,)
    
    # Check L2 normalization
    norm_face = np.linalg.norm(feats["face_embedding"])
    assert pytest.approx(norm_face, abs=1e-5) == 1.0


def test_ema_centroid_updating_and_prototypes():
    tracker = IdentityCentroidTracker(ema_alpha=0.95, max_prototypes=5, match_threshold=0.80)
    extractor = HybridPersonIDExtractor()
    
    vec1 = extractor.extract_face_embedding(identity_hint="ALICE")
    tracker.register_identity("ID_ALICE", vec1, name="Alice")
    
    assert "ID_ALICE" in tracker.identities
    assert len(tracker.identities["ID_ALICE"]) == 1

    # Slightly noisy vector -> should update prototype via EMA
    vec1_noisy = vec1 + np.random.normal(0, 0.02, 512).astype(np.float32)
    tracker.update_identity("ID_ALICE", vec1_noisy)
    
    matched_id, sim = tracker.match_embedding(vec1_noisy)
    assert matched_id == "ID_ALICE"
    assert sim > 0.85


def test_vector_db_hard_negative_exclusion():
    vdb = VectorDBManager(dim=512, similarity_threshold=0.80)
    
    # Create two close vectors
    vec_john = np.random.randn(512).astype(np.float32)
    vec_john /= np.linalg.norm(vec_john)
    
    vec_david = 0.88 * vec_john + 0.12 * np.random.randn(512).astype(np.float32)
    vec_david /= np.linalg.norm(vec_david)
    
    vdb.insert_vector("ID_JOHN", vec_john)
    vdb.insert_vector("ID_DAVID", vec_david)
    
    # Add Hard Negative Exclusion rule
    vdb.add_hard_negative("ID_JOHN", "ID_DAVID")
    assert vdb.is_blacklisted("ID_JOHN", "ID_DAVID") is True

    matched, sim, reason = vdb.verify_match_with_exclusion(vec_john, target_identity_id="ID_DAVID", query_identity_claim="ID_DAVID")
    assert "Hard Negative Blacklist" in reason or not matched


def test_vector_db_scale_benchmark():
    vdb = VectorDBManager()
    bench = vdb.benchmark_latency_and_memory(1_000_000)
    
    assert bench["scale_identities"] == 1_000_000
    assert bench["latency_ms"] < 30.0
    assert bench["ram_savings_factor"] == 4.0
