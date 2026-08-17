"""
Unit Tests for Modular Object Detector Module
"""

import pytest
import numpy as np
from object_intelligence.modular_object_detector import ModularHierarchicalObjectDetector, OpenSetMetricLearningRetriever


def test_coarse_and_fine_detection_pipeline():
    detector = ModularHierarchicalObjectDetector()
    results = detector.process_frame(tenant_id="TENANT_GLOBAL")
    
    assert len(results) > 0
    for r in results:
        assert "super_class" in r
        assert "fine_class" in r
        assert "combined_confidence" in r


def test_zero_downtime_custom_class_registration():
    retriever = OpenSetMetricLearningRetriever(vector_dim=128)
    
    proto = np.random.randn(128).astype(np.float32)
    reg_info = retriever.register_custom_class(
        class_name="custom_drone_quadcopter",
        super_class="machinery",
        prototype_vectors=[proto],
        tenant_id="TENANT_LOGISTICS"
    )
    
    assert reg_info["status"] == "ACTIVE"
    assert "custom_drone_quadcopter" in retriever.class_registry

    # Classify crop close to prototype
    matched_cls, score = retriever.classify_crop(proto, super_class_filter="machinery", tenant_id="TENANT_LOGISTICS")
    assert matched_cls == "custom_drone_quadcopter"
    assert score > 0.95
