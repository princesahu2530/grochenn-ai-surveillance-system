"""
Unit Tests for Canary Rollout & Model Compression Module
"""

import pytest
import numpy as np
from object_intelligence.canary_rollout_manager import CanaryRolloutManager, DynamicFeatureFlagRouter
from object_intelligence.model_compression import ModelCompressionOptimizer, DetectionCachingSafetyEngine


def test_dynamic_feature_flag_routing_and_canary():
    manager = CanaryRolloutManager()
    
    # Default baseline version
    ver1 = manager.process_customer_request("Customer_X")
    assert ver1 in [DynamicFeatureFlagRouter.DEFAULT_BASELINE, DynamicFeatureFlagRouter.NEW_CANDIDATE]

    # Explicit canary rollout
    manager.deploy_canary_to_customer("Customer_Y")
    ver_y = manager.router.get_model_version_for_customer("Customer_Y")
    assert ver_y == DynamicFeatureFlagRouter.NEW_CANDIDATE


def test_instant_rollback():
    manager = CanaryRolloutManager()
    manager.deploy_canary_to_customer("Customer_Z")
    
    rb_record = manager.rollback_engine.execute_instant_rollback("Customer_Z", reason="FP Spike")
    assert rb_record["new_version"] == DynamicFeatureFlagRouter.DEFAULT_BASELINE
    assert rb_record["execution_duration_ms"] < 5000.0
    assert manager.router.get_model_version_for_customer("Customer_Z") == DynamicFeatureFlagRouter.DEFAULT_BASELINE


def test_safety_zero_cache_enforcement():
    cache_engine = DetectionCachingSafetyEngine()
    
    # Safety event FALL_DETECTION -> must enforce fresh GPU evaluation (0-cache)
    dets, src = cache_engine.get_cached_or_evaluate("CAM_01", "FALL_DETECTION", 1, [{"bbox": (10, 10, 50, 50)}])
    assert "SAFETY_MANDATORY" in src

    # Standard non-safety event -> can use optical flow cache
    dets2, src2 = cache_engine.get_cached_or_evaluate("CAM_02", "LOITERING", 2, [{"bbox": (10, 10, 50, 50)}])
    dets3, src3 = cache_engine.get_cached_or_evaluate("CAM_02", "LOITERING", 3, [{"bbox": (10, 10, 50, 50)}])
    assert src3 == "CACHED_OPTICAL_FLOW"


def test_cost_reduction_matrix():
    optimizer = ModelCompressionOptimizer()
    report = optimizer.calculate_cost_reduction_matrix(5000)
    
    assert report["total_cameras"] == 5000
    assert report["total_cost_reduction_pct"] > 65.0
    assert report["monthly_fleet_savings_usd"] > 30000.0
