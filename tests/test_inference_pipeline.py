import time
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_ai.inference_pipeline import SpatialPolygonMask, TemporalDebouncer, EdgeAIPipeline

def test_spatial_polygon_mask():
    polygon = [(0, 0), (100, 0), (100, 100), (0, 100)]
    mask = SpatialPolygonMask(polygon)

    # Point inside polygon feet at (50, 50)
    inside_bbox = (20, 20, 80, 50)
    assert mask.is_inside(inside_bbox) is True

    # Point outside polygon feet at (150, 150)
    outside_bbox = (120, 120, 180, 150)
    assert mask.is_inside(outside_bbox) is False

def test_temporal_debouncer():
    debouncer = TemporalDebouncer(required_duration_sec=2.0)
    t0 = time.time()

    # First detection at t0 -> Should return False (not duration met yet)
    assert debouncer.process_detection("Track_01", t0) is False

    # Second detection at t0 + 1s -> Should return False
    assert debouncer.process_detection("Track_01", t0 + 1.0) is False

    # Third detection at t0 + 2.1s -> Should return True (Validated)
    assert debouncer.process_detection("Track_01", t0 + 2.1) is True

    # Subsequent detection when already triggered -> Should return False
    assert debouncer.process_detection("Track_01", t0 + 2.5) is False

def test_edge_ai_pipeline_subsampling():
    polygon = [(0, 0), (200, 0), (200, 200), (0, 200)]
    pipeline = EdgeAIPipeline("CAM_TEST", polygon)
    
    # Process 4 frames at 25 fps with target 5 fps (subsample = 5)
    # Frame 1 to 4 should be skipped (return None)
    for _ in range(4):
        res = pipeline.process_frame(None, raw_fps=25.0, target_fps=5.0)
        assert res is None
