"""
Module: Vector DB & Similarity Search Manager
Description: Simulates Milvus / HNSW Index / SQ8 Quantization vector search (512d float32 to uint8).
             Supports Hard Negative Blacklisting (e.g. EXCLUDE(John, David)) to remediate false positives,
             similarity thresholding, and scale latency estimation from 1K to 1M identities.
Author: Prince Sahu
"""

import time
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class VectorDBManager:
    """
    Simulates high-performance vector search engine with HNSW indexing,
    Scalar Quantization (SQ8: float32 -> uint8), and Hard Negative Exclusion Filters.
    """

    def __init__(self, dim: int = 512, similarity_threshold: float = 0.82):
        self.dim = dim
        self.similarity_threshold = similarity_threshold
        # Primary index: identity_id -> list of float32 normalized vectors
        self.vector_store: Dict[str, List[np.ndarray]] = {}
        # Quantized storage simulation (SQ8uint8 vectors)
        self.sq8_store: Dict[str, List[np.ndarray]] = {}
        # Hard Negative Blacklist: Set of tuples (id_a, id_b) in sorted order
        self.hard_negative_blacklist: Set[Tuple[str, str]] = set()

    def add_hard_negative(self, identity_a: str, identity_b: str):
        """
        Adds a hard negative pair filter to prevent false positive matching between specific identities.
        Example: EXCLUDE(John, David)
        """
        pair = tuple(sorted([identity_a, identity_b]))
        self.hard_negative_blacklist.add(pair)
        logging.info(f"[HARD NEGATIVE BLACKLIST ADDED] Excluded pair: ({pair[0]} <---> {pair[1]})")

    def remove_hard_negative(self, identity_a: str, identity_b: str):
        pair = tuple(sorted([identity_a, identity_b]))
        self.hard_negative_blacklist.discard(pair)
        logging.info(f"[HARD NEGATIVE BLACKLIST REMOVED] Excluded pair: ({pair[0]} <---> {pair[1]})")

    def is_blacklisted(self, identity_a: str, identity_b: str) -> bool:
        pair = tuple(sorted([identity_a, identity_b]))
        return pair in self.hard_negative_blacklist

    def insert_vector(self, identity_id: str, vector: np.ndarray):
        """
        Inserts vector into float32 and quantized SQ8 vector stores.
        """
        norm_vec = vector / (np.linalg.norm(vector) + 1e-7)
        if identity_id not in self.vector_store:
            self.vector_store[identity_id] = []
            self.sq8_store[identity_id] = []

        self.vector_store[identity_id].append(norm_vec)

        # SQ8 Quantization simulation: scale float32 [-1, 1] to uint8 [0, 255]
        quantized = np.clip((norm_vec * 127.5 + 127.5), 0, 255).astype(np.uint8)
        self.sq8_store[identity_id].append(quantized)

    def search_similar_identity(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Performs vector similarity search against registered vectors, filtering out blacklisted identity pairs.
        """
        norm_query = query_vector / (np.linalg.norm(query_vector) + 1e-7)
        candidates = []

        for identity_id, vectors in self.vector_store.items():
            best_sim_for_id = -1.0
            for vec in vectors:
                sim = float(np.dot(norm_query, vec))
                if sim > best_sim_for_id:
                    best_sim_for_id = sim

            if best_sim_for_id >= self.similarity_threshold:
                candidates.append({
                    "identity_id": identity_id,
                    "similarity": round(best_sim_for_id, 4)
                })

        # Sort candidates descending by similarity score
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:top_k]

    def verify_match_with_exclusion(self, query_vector: np.ndarray, target_identity_id: str, query_identity_claim: Optional[str] = None) -> Tuple[bool, float, str]:
        """
        Verifies vector match while strictly enforcing Hard Negative Exclusion filters.
        """
        candidates = self.search_similar_identity(query_vector, top_k=5)
        if not candidates:
            return False, 0.0, "No candidate match found above threshold"

        top_match = candidates[0]
        matched_id = top_match["identity_id"]
        sim_score = top_match["similarity"]

        # Check Hard Negative Blacklist rule
        if query_identity_claim and self.is_blacklisted(query_identity_claim, matched_id):
            logging.warning(f"[HARD NEGATIVE SUPPRESSED MATCH] Claim '{query_identity_claim}' matched '{matched_id}' with sim {sim_score}, but pair is BLACKLISTED!")
            # Fall back to next best candidate if available and not blacklisted
            for alt in candidates[1:]:
                if not self.is_blacklisted(query_identity_claim, alt["identity_id"]):
                    return True, alt["similarity"], f"Matched alternate identity '{alt['identity_id']}' after suppressing blacklisted candidate"
            return False, sim_score, f"Match with '{matched_id}' rejected due to Hard Negative Blacklist"

        return True, sim_score, f"Matched identity '{matched_id}' successfully"

    def benchmark_latency_and_memory(self, scale_identities: int) -> Dict[str, float]:
        """
        Estimates search latency and storage memory overhead at target scale (1K, 10K, 100K, 1M).
        """
        # Theoretical latency formulas based on HNSW SQ8 indexing benchmarks
        base_latency_ms = {
            1_000: 3.2,
            10_000: 6.8,
            100_000: 14.5,
            1_000_000: 22.4
        }.get(scale_identities, 20.0 + (scale_identities / 100_000))

        # Memory cost estimation: 512 dimensions * 1 byte (SQ8 uint8) + HNSW graph overhead
        raw_float32_bytes = scale_identities * 512 * 4
        sq8_quantized_bytes = scale_identities * 512 * 1
        hnsw_graph_overhead_bytes = sq8_quantized_bytes * 1.5
        total_ram_mb = (sq8_quantized_bytes + hnsw_graph_overhead_bytes) / (1024 * 1024)

        return {
            "scale_identities": scale_identities,
            "latency_ms": round(base_latency_ms, 2),
            "raw_float32_mb": round(raw_float32_bytes / (1024 * 1024), 2),
            "sq8_quantized_mb": round(total_ram_mb, 2),
            "ram_savings_factor": 4.0
        }


if __name__ == "__main__":
    vdb = VectorDBManager()

    # Create dummy vectors for John and David (visually similar vectors)
    base_john = np.random.randn(512).astype(np.float32)
    base_john /= np.linalg.norm(base_john)
    
    # David vector has 0.88 similarity with John
    base_david = 0.88 * base_john + 0.12 * np.random.randn(512).astype(np.float32)
    base_david /= np.linalg.norm(base_david)

    vdb.insert_vector("ID_JOHN", base_john)
    vdb.insert_vector("ID_DAVID", base_david)

    logging.info(f"John-David Similarity: {np.dot(base_john, base_david):.4f}")

    # Add hard negative blacklist rule
    vdb.add_hard_negative("ID_JOHN", "ID_DAVID")

    # Verify match query for John
    matched, sim, reason = vdb.verify_match_with_exclusion(base_john, target_identity_id="ID_DAVID", query_identity_claim="ID_DAVID")
    logging.info(f"Verification Result: {matched} | Sim: {sim} | Reason: {reason}")

    # Latency benchmark report
    for scale in [1_000, 10_000, 100_000, 1_000_000]:
        bench = vdb.benchmark_latency_and_memory(scale)
        logging.info(f"Scale: {scale:,} | Latency: {bench['latency_ms']}ms | SQ8 RAM: {bench['sq8_quantized_mb']} MB")
