"""
Module: Edge AI Inference Pipeline
Description: Subsamples 25 FPS down to 5 FPS, executes Object Detection & ByteTrack tracking,
             evaluates Spatial Polygon intersection, and applies Temporal Debouncing.
Author: Prince Sahu
"""

import time
import logging
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SpatialPolygonMask:
    def __init__(self, polygon_coords: List[Tuple[int, int]]):
        self.polygon_coords = polygon_coords

    def is_inside(self, bbox: Tuple[int, int, int, int]) -> bool:
        # BBox format: (x_min, y_min, x_max, y_max)
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = bbox[3] # Ground level point (feet of the person)
        
        # Simple Ray-Casting Point-in-Polygon Check
        n = len(self.polygon_coords)
        inside = False
        p1x, p1y = self.polygon_coords[0]
        for i in range(n + 1):
            p2x, p2y = self.polygon_coords[i % n]
            if center_y > min(p1y, p2y):
                if center_y <= max(p1y, p2y):
                    if center_x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (center_y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or center_x <= xinters:
                                inside = not inside
            p1x, p1y = p2x, p2y
        return inside

class TemporalDebouncer:
    """
    Ensures an event persists continuously over a required time duration (e.g. 3 seconds)
    before validating and triggering an alert.
    """
    def __init__(self, required_duration_sec: float = 3.0, timeout_sec: float = 2.0):
        self.required_duration_sec = required_duration_sec
        self.timeout_sec = timeout_sec
        self.active_tracks: Dict[str, dict] = {}

    def process_detection(self, track_id: str, current_time: float) -> bool:
        if track_id not in self.active_tracks:
            self.active_tracks[track_id] = {
                'first_seen': current_time,
                'last_seen': current_time,
                'triggered': False
            }
            return False

        track = self.active_tracks[track_id]
        track['last_seen'] = current_time
        duration = current_time - track['first_seen']

        if duration >= self.required_duration_sec and not track['triggered']:
            track['triggered'] = True
            return True # EVENT VALIDATED

        return False

    def cleanup_stale(self, current_time: float):
        stale_keys = [tid for tid, data in self.active_tracks.items() if (current_time - data['last_seen']) > self.timeout_sec]
        for tid in stale_keys:
            del self.active_tracks[tid]

class EdgeAIPipeline:
    def __init__(self, camera_id: str, polygon_coords: List[Tuple[int, int]]):
        self.camera_id = camera_id
        self.spatial_mask = SpatialPolygonMask(polygon_coords)
        self.debouncer = TemporalDebouncer(required_duration_sec=3.0)
        self.frame_counter = 0

    def process_frame(self, frame_data=None, detections: List[dict] = None, raw_fps: float = 25.0, target_fps: float = 5.0):
        self.frame_counter += 1
        sample_interval = max(1, int(raw_fps / target_fps))

        # 1. Frame Subsampling (Process 1 in every 5 frames)
        if self.frame_counter % sample_interval != 0:
            return None

        current_time = time.time()
        
        # 2. Simulated / Provided YOLOv8 + ByteTrack Detections
        if detections is None:
            detections = [
                {'track_id': 'Track_101', 'class': 'person', 'confidence': 0.89, 'bbox': (150, 200, 250, 500)}
            ]

        validated_events = []
        for det in detections:
            # 3. Spatial Polygon Filtering
            if self.spatial_mask.is_inside(det['bbox']):
                # 4. Temporal Debounce Check
                is_alert_validated = self.debouncer.process_detection(det['track_id'], current_time)
                if is_alert_validated:
                    logging.info(f"[VALIDATED ALERT] Camera: {self.camera_id} | Track: {det['track_id']} | Type: Restricted Intrusion")
                    event = {
                        'camera_id': self.camera_id,
                        'track_id': det['track_id'],
                        'event_type': 'UNAUTHORIZED_INTRUSION',
                        'confidence': det['confidence'],
                        'spatial_validated': True,
                        'persistence_sec': 3.5,
                        'timestamp': current_time
                    }
                    validated_events.append(event)

        self.debouncer.cleanup_stale(current_time)
        return validated_events[0] if validated_events else None

if __name__ == "__main__":
    polygon = [(100, 100), (400, 100), (400, 600), (100, 600)]
    pipeline = EdgeAIPipeline("CAM_STORE_01", polygon)
    logging.info("Edge AI Pipeline initialized successfully. Testing debouncer loop...")
    start = time.time()
    for t in range(25): # simulate 5 seconds at 5fps interval
        res = pipeline.process_frame(None, raw_fps=5.0, target_fps=5.0)
        if res:
            logging.info(f"Triggered validated event after {time.time() - start:.2f}s: {res}")
            break
        time.sleep(0.2)

