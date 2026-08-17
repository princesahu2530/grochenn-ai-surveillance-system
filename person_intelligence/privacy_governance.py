"""
Module: Privacy Governance, Consent Management & Audit Trail
Description: Provides Edge Anonymization (Bloom Filter consent checking & face blurring),
             Right to Be Forgotten (RTBF) API, Immutable ClickHouse Audit Logger with SHA256 hashing,
             and 30-day automated time-based TTL data pruning.
Author: Prince Sahu
"""

import time
import hashlib
import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class EdgeConsentBloomFilter:
    """
    Simulates lightweight Edge Bloom Filter checking consent status of subject IDs.
    Returns True if subject has explicit consent for tracking; False if non-consented/opted-out.
    """

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        # Set of hashed non-consented / opted-out identity tokens
        self.opt_out_hashes: Set[str] = set()

    def _hash_token(self, subject_token: str) -> str:
        return hashlib.sha256(subject_token.encode('utf-8')).hexdigest()[:16]

    def register_opt_out(self, subject_token: str):
        token_hash = self._hash_token(subject_token)
        self.opt_out_hashes.add(token_hash)
        logging.info(f"[PRIVACY OPT-OUT REGISTERED] Hash: {token_hash} (Subject: {subject_token})")

    def revoke_opt_out(self, subject_token: str):
        token_hash = self._hash_token(subject_token)
        self.opt_out_hashes.discard(token_hash)
        logging.info(f"[PRIVACY OPT-OUT REVOKED] Hash: {token_hash}")

    def is_consented(self, subject_token: str) -> bool:
        token_hash = self._hash_token(subject_token)
        return token_hash not in self.opt_out_hashes


class EdgeAnonymizationEngine:
    """
    Applies real-time Gaussian blurring or pixelation to non-consented face bounding boxes on edge video frames.
    """

    def __init__(self, consent_filter: EdgeConsentBloomFilter):
        self.consent_filter = consent_filter

    def anonymize_frame(self, frame: np.ndarray, detections: List[Dict]) -> Tuple[np.ndarray, int]:
        """
        Processes frame and blurs non-consented facial regions in-place.
        Returns (anonymized_frame, blurred_count).
        """
        anonymized = frame.copy()
        blurred_count = 0

        for det in detections:
            subject_token = det.get("subject_token", "UNKNOWN_GUEST")
            bbox = det.get("bbox", (0, 0, 10, 10))

            if not self.consent_filter.is_consented(subject_token):
                x1, y1, x2, y2 = bbox
                # Boundary safety clamping
                h, w, _ = frame.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    crop = anonymized[y1:y2, x1:x2]
                    # Apply heavy Gaussian blur to anonymize face
                    k_w = max(15, (x2 - x1) // 2 | 1)
                    k_h = max(15, (y2 - y1) // 2 | 1)
                    blurred_crop = cv2.GaussianBlur(crop, (k_w, k_h), 30)
                    anonymized[y1:y2, x1:x2] = blurred_crop
                    blurred_count += 1

        return anonymized, blurred_count


class ClickHouseAuditLogger:
    """
    Simulates ClickHouse immutable audit logging engine (surveillance.audit.access).
    Uses cryptographic hash chaining (SHA-256 block chain) to ensure tamper-proof audit trails.
    """

    def __init__(self):
        self.audit_records: List[Dict] = []
        self.last_block_hash: str = "0000000000000000000000000000000000000000000000000000000000000000"

    def log_access_event(self, operator_id: str, action: str, target_identity_id: str, details: str = "") -> Dict:
        timestamp = time.time()
        record_id = f"AUDIT_{len(self.audit_records) + 1:06d}"
        
        payload = f"{record_id}|{timestamp:.4f}|{operator_id}|{action}|{target_identity_id}|{details}|{self.last_block_hash}"
        block_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()

        audit_entry = {
            "record_id": record_id,
            "timestamp": timestamp,
            "operator_id": operator_id,
            "action": action,
            "target_identity_id": target_identity_id,
            "details": details,
            "prev_hash": self.last_block_hash,
            "block_hash": block_hash
        }

        self.last_block_hash = block_hash
        self.audit_records.append(audit_entry)
        logging.info(f"[CLICKHOUSE AUDIT LOG] {record_id} | Op: {operator_id} | Action: {action} | Target: {target_identity_id}")
        return audit_entry

    def verify_integrity(self) -> bool:
        """
        Validates cryptographic hash chain across all audit entries.
        """
        prev_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        for rec in self.audit_records:
            payload = f"{rec['record_id']}|{rec['timestamp']:.4f}|{rec['operator_id']}|{rec['action']}|{rec['target_identity_id']}|{rec['details']}|{prev_hash}"
            expected_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()
            if rec["block_hash"] != expected_hash:
                logging.error(f"[AUDIT CHAIN CORRUPTED] Record {rec['record_id']} hash mismatch!")
                return False
            prev_hash = rec["block_hash"]
        return True

    def query_logs_by_operator(self, operator_id: str) -> List[Dict]:
        return [r for r in self.audit_records if r["operator_id"] == operator_id]


class PrivacyGovernanceManager:
    """
    Unified manager handling Right to Be Forgotten (RTBF), consent enforcement, audit logging,
    and 30-day time-based TTL pruner.
    """

    def __init__(self, retention_days: int = 30):
        self.retention_seconds = retention_days * 24 * 3600
        self.consent_filter = EdgeConsentBloomFilter()
        self.anonymizer = EdgeAnonymizationEngine(self.consent_filter)
        self.audit_logger = ClickHouseAuditLogger()
        # Simulated transient database of identity captures: { capture_id: dict }
        self.transient_captures: Dict[str, Dict] = {}

    def store_transient_capture(self, capture_id: str, identity_id: str, metadata: dict) -> Dict:
        now = time.time()
        record = {
            "capture_id": capture_id,
            "identity_id": identity_id,
            "timestamp": now,
            "ttl_expiration": now + self.retention_seconds,
            "metadata": metadata
        }
        self.transient_captures[capture_id] = record
        return record

    def execute_rtbf_deletion(self, identity_id: str, operator_id: str) -> Dict[str, int]:
        """
        Executes Right to Be Forgotten (RTBF) API (DELETE /v1/identities/{id}).
        Purges all transient captures, updates consent opt-out filter, and logs to ClickHouse audit trail.
        """
        purged_captures = 0
        to_delete = [cid for cid, cap in self.transient_captures.items() if cap["identity_id"] == identity_id]
        for cid in to_delete:
            del self.transient_captures[cid]
            purged_captures += 1

        # Register subject opt-out to suppress future tracking
        self.consent_filter.register_opt_out(identity_id)

        # Log immutable audit entry
        self.audit_logger.log_access_event(
            operator_id=operator_id,
            action="RTBF_PURGE_DELETE",
            target_identity_id=identity_id,
            details=f"Purged {purged_captures} transient captures and registered consent opt-out"
        )

        return {
            "identity_id": identity_id,
            "purged_captures": purged_captures,
            "status": "PURGED_AND_OPTED_OUT"
        }

    def run_ttl_pruner(self, current_time: Optional[float] = None) -> int:
        """
        Purges transient captures older than the 30-day retention TTL threshold.
        """
        if current_time is None:
            current_time = time.time()

        stale_ids = [cid for cid, cap in self.transient_captures.items() if current_time >= cap["ttl_expiration"]]
        for cid in stale_ids:
            del self.transient_captures[cid]

        if stale_ids:
            logging.info(f"[30-DAY TTL PRUNER] Automatically purged {len(stale_ids)} expired transient captures")
        return len(stale_ids)


if __name__ == "__main__":
    gov = PrivacyGovernanceManager(retention_days=30)

    # Test transient capture storage
    gov.store_transient_capture("CAP_001", "ID_USER_42", {"location": "Cam_North"})
    gov.store_transient_capture("CAP_002", "ID_USER_42", {"location": "Cam_South"})

    # Execute RTBF deletion
    res = gov.execute_rtbf_deletion("ID_USER_42", operator_id="ADMIN_ALICE")
    logging.info(f"RTBF Result: {res}")

    # Verify audit log integrity
    is_valid = gov.audit_logger.verify_integrity()
    logging.info(f"Audit Log Integrity Valid: {is_valid}")
