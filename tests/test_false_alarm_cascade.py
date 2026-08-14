import time
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloud_processing.false_alarm_cascade import FalseAlarmCascadeEngine

def test_false_alarm_cascade_pass():
    engine = FalseAlarmCascadeEngine()
    valid_payload = {
        'camera_id': 'CAM_01',
        'event_type': 'UNAUTHORIZED_INTRUSION',
        'confidence': 0.92,
        'spatial_validated': True,
        'persistence_sec': 3.5,
        'timestamp': time.time()
    }
    assert engine.process_event(valid_payload) is True

def test_false_alarm_cascade_low_score_suppressed():
    engine = FalseAlarmCascadeEngine()
    low_score_payload = {
        'camera_id': 'CAM_01',
        'event_type': 'UNAUTHORIZED_INTRUSION',
        'confidence': 0.70, # Below 0.88 threshold
        'spatial_validated': True,
        'persistence_sec': 3.5,
        'timestamp': time.time()
    }
    assert engine.process_event(low_score_payload) is False

def test_false_alarm_cascade_short_duration_suppressed():
    engine = FalseAlarmCascadeEngine()
    short_duration_payload = {
        'camera_id': 'CAM_01',
        'event_type': 'UNAUTHORIZED_INTRUSION',
        'confidence': 0.90,
        'spatial_validated': True,
        'persistence_sec': 1.5, # Below 3.0 seconds
        'timestamp': time.time()
    }
    assert engine.process_event(short_duration_payload) is False
