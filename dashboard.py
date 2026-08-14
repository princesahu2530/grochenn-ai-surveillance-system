"""
Module: Interactive Live Web Monitoring Dashboard
Description: FastAPI web application serving live surveillance telemetry, synthetic video stream, 
             and 4-stage false alarm cascade logs.
Author: Prince Sahu
"""

import os
import sys
import time
import random
import cv2
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Enable UTF-8 encoding for Windows terminal output
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from edge_ingestion.rtsp_ring_buffer import RingBufferManager
from cloud_processing.false_alarm_cascade import FalseAlarmCascadeEngine

app = FastAPI(title="Enterprise AI CCTV & Surveillance System Dashboard")

# Static files directory
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
os.makedirs(WEB_DIR, exist_ok=True)

# Shared memory log buffer for events
recent_events = []

def generate_mock_events():
    event_types = ['UNAUTHORIZED_INTRUSION', 'FALL_DETECTION', 'LOITERING']
    cams = ['CAM_STORE_NORTH_01', 'CAM_STORE_SOUTH_04', 'CAM_WAREHOUSE_02']
    actions = ['DISPATCHED', 'SUPPRESSED']
    
    for i in range(8):
        action = 'DISPATCHED' if random.random() > 0.3 else 'SUPPRESSED'
        recent_events.append({
            'timestamp': time.time() - (i * 12),
            'camera_id': random.choice(cams),
            'event_type': random.choice(event_types),
            'confidence': round(random.uniform(0.72, 0.98), 2),
            'action': action
        })

generate_mock_events()

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return HTMLResponse("<h1>Dashboard HTML not found</h1>")

@app.get("/api/stats")
def get_stats():
    return {
        "camera_count": 5000,
        "total_max_cameras": 10000,
        "baseline_cost_per_cam": 41.45,
        "optimized_cost_per_cam": 6.38,
        "cost_reduction_pct": 84.6,
        "false_alarm_reduction_pct": 88.5,
        "ring_buffer_disk_usage_pct": 42.8 + random.uniform(-1.5, 1.5)
    }

@app.get("/api/events")
def get_events():
    # Occasionally append a new event
    if random.random() > 0.6:
        event_types = ['UNAUTHORIZED_INTRUSION', 'FALL_DETECTION', 'LOITERING']
        cams = ['CAM_STORE_NORTH_01', 'CAM_STORE_SOUTH_04', 'CAM_WAREHOUSE_02']
        action = 'DISPATCHED' if random.random() > 0.35 else 'SUPPRESSED'
        recent_events.insert(0, {
            'timestamp': time.time(),
            'camera_id': random.choice(cams),
            'event_type': random.choice(event_types),
            'confidence': round(random.uniform(0.75, 0.99), 2),
            'action': action
        })
        if len(recent_events) > 20:
            recent_events.pop()
    return recent_events[:10]

@app.get("/api/stream_frame")
def get_stream_frame():
    # Create synthetic frame with bounding box and polygon mask overlay
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Background grid design
    for y in range(0, 480, 40):
        cv2.line(frame, (0, y), (640, y), (20, 25, 35), 1)
    for x in range(0, 640, 40):
        cv2.line(frame, (x, 0), (x, 480), (20, 25, 35), 1)

    # Polygon ROI Mask (Restricted zone)
    poly = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], np.int32)
    poly_overlay = frame.copy()
    cv2.fillPoly(poly_overlay, [poly], (0, 100, 255))
    cv2.addWeighted(poly_overlay, 0.2, frame, 0.8, 0, frame)
    cv2.polylines(frame, [poly], True, (0, 165, 255), 2)

    # Animated person bbox
    t = time.time()
    cx = int(200 + np.sin(t * 2) * 120)
    cy = int(220 + np.cos(t * 1.5) * 40)
    
    # Check polygon intersection
    inside_roi = (100 <= cx <= 500) and (100 <= (cy + 100) <= 400)
    box_color = (0, 0, 255) if inside_roi else (0, 255, 0)
    label = "INTRUSION DETECTED (0.94)" if inside_roi else "PERSON (0.91)"

    cv2.rectangle(frame, (cx, cy), (cx + 80, cy + 120), box_color, 2)
    cv2.rectangle(frame, (cx, cy - 22), (cx + 180, cy), box_color, -1)
    cv2.putText(frame, label, (cx + 4, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # HUD Text
    cv2.putText(frame, "CAM_STORE_NORTH_01 | RESTRICTED ZONE A", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "STATUS: RECORDING (NVMe RING BUFFER)", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1)

    # Encode to JPEG
    _, encoded = cv2.imencode('.jpg', frame)
    return Response(content=encoded.tobytes(), media_type="image/jpeg")

def start_dashboard():
    print("=" * 80)
    print(" 🚀 STARTING ENTERPRISE AI SURVEILLANCE WEB DASHBOARD")
    print(" URL: http://localhost:8000")
    print("=" * 80)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    start_dashboard()
