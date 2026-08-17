"""
Unit Tests for Privacy Governance & Compliance Module
"""

import pytest
import numpy as np
import time
from person_intelligence.privacy_governance import PrivacyGovernanceManager, EdgeConsentBloomFilter, ClickHouseAuditLogger


def test_edge_consent_bloom_filter():
    bloom = EdgeConsentBloomFilter()
    assert bloom.is_consented("USER_123") is True
    
    bloom.register_opt_out("USER_123")
    assert bloom.is_consented("USER_123") is False

    bloom.revoke_opt_out("USER_123")
    assert bloom.is_consented("USER_123") is True


def test_clickhouse_audit_logger_hash_chain():
    logger = ClickHouseAuditLogger()
    rec1 = logger.log_access_event("OPERATOR_A", "SEARCH", "ID_101", "Query 1")
    rec2 = logger.log_access_event("OPERATOR_B", "RTBF_DELETE", "ID_102", "Purge 2")
    
    assert rec2["prev_hash"] == rec1["block_hash"]
    assert logger.verify_integrity() is True


def test_rtbf_deletion_and_ttl_pruning():
    gov = PrivacyGovernanceManager(retention_days=30)
    
    # Store transient capture
    gov.store_transient_capture("CAP_01", "ID_TARGET", {"cam": "CAM_01"})
    assert "CAP_01" in gov.transient_captures

    # Execute RTBF deletion
    res = gov.execute_rtbf_deletion("ID_TARGET", operator_id="ADMIN_ALICE")
    assert res["status"] == "PURGED_AND_OPTED_OUT"
    assert "CAP_01" not in gov.transient_captures
    assert gov.consent_filter.is_consented("ID_TARGET") is False

    # Store expired capture and run TTL pruner
    expired_time = time.time() + (31 * 24 * 3600)
    gov.store_transient_capture("CAP_EXPIRED", "ID_OTHER", {"cam": "CAM_02"})
    pruned_count = gov.run_ttl_pruner(current_time=expired_time)
    assert pruned_count >= 1
