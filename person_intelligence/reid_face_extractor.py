"""
Module: Person ReID & Facial Feature Extractor
Description: Implements hybrid ArcFace (512d) facial embeddings + OSNet (256d) Person ReID embeddings.
             Supports Exponential Moving Average (EMA) centroid updating for appearance drift
             and multi-prototype clustering (up to 5 centroids per identity).
Author: Prince Sahu
"""

import time
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class HybridPersonIDExtractor:
    """
    Extracts hybrid visual representation combining ArcFace (512-dim facial vector)
    and OSNet (256-dim body ReID vector).
    """

    def __init__(self, face_dim: int = 512, reid_dim: int = 256, seed: int = 42):
        self.face_dim = face_dim
        self.reid_dim = reid_dim
        np.random.seed(seed)

    def extract_face_embedding(self, face_crop: Optional[np.ndarray] = None, identity_hint: str = "") -> np.ndarray:
        """
        Simulates ArcFace 512d face feature extraction.
        Returns L2-normalized 512-dimensional vector.
        """
        # Deterministic simulation based on identity_hint if provided
        if identity_hint:
            seed_val = abs(hash(identity_hint)) % (2**31)
            rng = np.random.RandomState(seed_val)
            vec = rng.randn(self.face_dim).astype(np.float32)
        else:
            vec = np.random.randn(self.face_dim).astype(np.float32)
        
        # L2 Normalize
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-7)

    def extract_reid_embedding(self, body_crop: Optional[np.ndarray] = None, identity_hint: str = "") -> np.ndarray:
        """
        Simulates OSNet 256d person body ReID feature extraction.
        Returns L2-normalized 256-dimensional vector.
        """
        if identity_hint:
            seed_val = (abs(hash(identity_hint)) + 1007) % (2**31)
            rng = np.random.RandomState(seed_val)
            vec = rng.randn(self.reid_dim).astype(np.float32)
        else:
            vec = np.random.randn(self.reid_dim).astype(np.float32)
        
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-7)

    def extract_hybrid_features(self, frame_crop: Optional[np.ndarray] = None, identity_hint: str = "") -> Dict[str, np.ndarray]:
        """
        Extracts both facial (512d) and ReID (256d) feature embeddings.
        """
        face_emb = self.extract_face_embedding(frame_crop, identity_hint=identity_hint)
        reid_emb = self.extract_reid_embedding(frame_crop, identity_hint=identity_hint)
        return {
            "face_embedding": face_emb,
            "reid_embedding": reid_emb
        }


class IdentityCentroidTracker:
    """
    Tracks identity centroids with Exponential Moving Average (EMA) update for aging/appearance drift.
    Maintains multi-prototype clusters (up to max_prototypes per identity).
    """

    def __init__(self, ema_alpha: float = 0.95, max_prototypes: int = 5, match_threshold: float = 0.82):
        self.ema_alpha = ema_alpha
        self.max_prototypes = max_prototypes
        self.match_threshold = match_threshold
        # Schema: { identity_id: List[np.ndarray] } -> list of prototype centroids
        self.identities: Dict[str, List[np.ndarray]] = {}
        # Schema: { identity_id: dict of metadata }
        self.metadata: Dict[str, dict] = {}

    def register_identity(self, identity_id: str, embedding: np.ndarray, name: str = ""):
        """
        Initializes identity record with first prototype vector.
        """
        norm_emb = embedding / (np.linalg.norm(embedding) + 1e-7)
        self.identities[identity_id] = [norm_emb]
        self.metadata[identity_id] = {
            "name": name or identity_id,
            "created_at": time.time(),
            "updated_at": time.time(),
            "updates_count": 1
        }
        logging.info(f"[IDENTITY REGISTERED] ID: {identity_id} | Name: {name or identity_id}")

    def update_identity(self, identity_id: str, new_embedding: np.ndarray):
        """
        Applies Exponential Moving Average (EMA) updating to closest prototype or adds new prototype.
        Formula: C_t = 0.95 * C_{t-1} + 0.05 * E_{new}
        """
        if identity_id not in self.identities:
            self.register_identity(identity_id, new_embedding)
            return

        norm_new = new_embedding / (np.linalg.norm(new_embedding) + 1e-7)
        prototypes = self.identities[identity_id]

        # Find best matching prototype for this identity
        best_sim = -1.0
        best_idx = 0
        for idx, proto in enumerate(prototypes):
            sim = float(np.dot(proto, norm_new))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_sim >= self.match_threshold:
            # Update existing centroid via EMA
            old_proto = prototypes[best_idx]
            updated_proto = self.ema_alpha * old_proto + (1.0 - self.ema_alpha) * norm_new
            updated_proto = updated_proto / (np.linalg.norm(updated_proto) + 1e-7)
            prototypes[best_idx] = updated_proto
            self.metadata[identity_id]["updated_at"] = time.time()
            self.metadata[identity_id]["updates_count"] += 1
            logging.info(f"[EMA UPDATE] Identity '{identity_id}' (Proto #{best_idx}) updated. Sim: {best_sim:.4f}")
        else:
            # Appearance change / new lighting pose -> Add new prototype if room
            if len(prototypes) < self.max_prototypes:
                prototypes.append(norm_new)
                logging.info(f"[NEW PROTOTYPE] Identity '{identity_id}' added prototype #{len(prototypes)}")
            else:
                # Replace oldest prototype if max capacity reached
                prototypes.pop(0)
                prototypes.append(norm_new)
                logging.info(f"[PROTOTYPE ROTATION] Identity '{identity_id}' rotated oldest prototype.")

    def match_embedding(self, query_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Matches query embedding against all prototype centroids across identities.
        Returns (best_identity_id, max_similarity).
        """
        norm_query = query_embedding / (np.linalg.norm(query_embedding) + 1e-7)
        best_identity = None
        max_sim = -1.0

        for identity_id, prototypes in self.identities.items():
            for proto in prototypes:
                sim = float(np.dot(proto, norm_query))
                if sim > max_sim:
                    max_sim = sim
                    best_identity = identity_id

        if max_sim >= self.match_threshold:
            return best_identity, max_sim
        return None, max_sim


if __name__ == "__main__":
    extractor = HybridPersonIDExtractor()
    tracker = IdentityCentroidTracker()

    # Test feature extraction
    feats_john = extractor.extract_hybrid_features(identity_hint="John_Doe")
    tracker.register_identity("ID_101", feats_john["face_embedding"], name="John Doe")

    # Test EMA update with slightly noisy capture
    noisy_john = feats_john["face_embedding"] + np.random.normal(0, 0.05, 512).astype(np.float32)
    tracker.update_identity("ID_101", noisy_john)

    # Test matching
    matched_id, sim = tracker.match_embedding(noisy_john)
    logging.info(f"Match Result -> ID: {matched_id}, Cosine Similarity: {sim:.4f}")
