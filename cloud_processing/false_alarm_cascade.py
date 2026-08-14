"""
Module: Cloud 4-Stage False Alarm Reduction Cascade Engine
Description: Filters incoming edge triggers through Spatial, Temporal Persistence, 
             Dynamic Score Scaling, and Business Schedule Mask layers.
Author: Prince Sahu
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class FalseAlarmCascadeEngine:
    def __init__(self):
        # Dynamic Confidence Thresholds per event type
        self.thresholds = {
            'FALL_DETECTION': 0.70,        # High Recall Priority (Missed fall is dangerous)
            'UNAUTHORIZED_INTRUSION': 0.88, # High Precision Priority (Prevents guard fatigue)
            'LOITERING': 0.80
        }
        # Business Operating Hours (9 AM to 6 PM)
        self.business_hours_start = 9
        self.business_hours_end = 18

    def process_event(self, event_payload: Dict[str, Any]) -> bool:
        """
        Returns True if alert is confirmed valid; False if suppressed as false positive.
        """
        event_type = event_payload.get('event_type', 'UNAUTHORIZED_INTRUSION')
        confidence = event_payload.get('confidence', 0.0)
        timestamp = event_payload.get('timestamp', time.time())

        # Stage 1: Spatial Polygon Check (Pre-validated at edge)
        if not event_payload.get('spatial_validated', True):
            logging.info(f"[SUPPRESSED] Stage 1 Fail: Event outside ground polygon.")
            return False

        # Stage 2: Temporal Persistence Check (Pre-validated at edge)
        if event_payload.get('persistence_sec', 3.5) < 3.0:
            logging.info(f"[SUPPRESSED] Stage 2 Fail: Duration < 3.0 seconds.")
            return False

        # Stage 3: Dynamic Score Scaling
        required_score = self.thresholds.get(event_type, 0.85)
        if confidence < required_score:
            logging.info(f"[SUPPRESSED] Stage 3 Fail: Score {confidence:.2f} < Threshold {required_score:.2f}")
            return False

        # Stage 4: Business Schedule / Mask Filter
        event_hour = datetime.fromtimestamp(timestamp).hour
        if event_type == 'LOITERING' and (self.business_hours_start <= event_hour < self.business_hours_end):
            logging.info(f"[SUPPRESSED] Stage 4 Fail: Loitering during authorized operating hours ({event_hour}:00).")
            return False

        logging.info(f"✅ [ALERT DISPATCHED] Event: {event_type} | Camera: {event_payload['camera_id']} | Score: {confidence:.2f}")
        return True

if __name__ == "__main__":
    cascade = FalseAlarmCascadeEngine()
    test_payload = {
        'camera_id': 'CAM_STORE_01',
        'event_type': 'UNAUTHORIZED_INTRUSION',
        'confidence': 0.91,
        'spatial_validated': True,
        'persistence_sec': 3.5,
        'timestamp': time.time()
    }
    cascade.process_event(test_payload)
