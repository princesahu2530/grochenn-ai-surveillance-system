import os
import shutil
import tempfile
import time
import pytest
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge_ingestion.rtsp_ring_buffer import RingBufferManager, RTSPIngestor

def test_ring_buffer_manager_creation():
    temp_dir = tempfile.mkdtemp()
    try:
        buffer = RingBufferManager(storage_dir=temp_dir, max_disk_usage_pct=90.0)
        assert os.path.exists(temp_dir)
        usage = buffer.get_disk_usage_percent()
        assert 0.0 <= usage <= 100.0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_synthetic_rtsp_ingestor():
    temp_dir = tempfile.mkdtemp()
    try:
        buffer = RingBufferManager(storage_dir=temp_dir, max_disk_usage_pct=90.0, chunk_duration_sec=2)
        ingestor = RTSPIngestor("TEST_CAM", "rtsp://localhost:8554/live", buffer, simulate=True)
        ingestor.start()
        time.sleep(2.5)
        ingestor.is_running = False
        
        # Check if video file chunk was created
        files = os.listdir(temp_dir)
        mp4_files = [f for f in files if f.endswith('.mp4')]
        assert len(mp4_files) >= 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
