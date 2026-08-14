"""
Module: Edge Ingestion & Disk Ring Buffer Manager
Description: Resilient RTSP Stream Ingestion with Hardware Decoding, Exponential Backoff, 
             60-second .mp4 chunking, and NVMe Disk Auto-Pruner (Threshold: 85%).
Author: Prince Sahu
"""

import os
import time
import cv2
import threading
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class RingBufferManager:
    def __init__(self, storage_dir="/var/surveillance/buffer", max_disk_usage_pct=85.0, chunk_duration_sec=60):
        self.storage_dir = storage_dir
        self.max_disk_usage_pct = max_disk_usage_pct
        self.chunk_duration_sec = chunk_duration_sec
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Start background disk pruner thread
        self._pruner_thread = threading.Thread(target=self._auto_prune_loop, daemon=True)
        self._pruner_thread.start()

    def get_disk_usage_percent(self):
        total, used, free = shutil.disk_usage(self.storage_dir)
        return (used / total) * 100.0

    def _auto_prune_loop(self):
        logging.info("RingBufferManager: Auto-pruning service initialized.")
        while True:
            try:
                usage = self.get_disk_usage_percent()
                if usage > self.max_disk_usage_pct:
                    logging.warning(f"[DISK FULL WARN] Usage at {usage:.2f}%. Pruning oldest chunks...")
                    mp4_files = sorted(
                        [os.path.join(self.storage_dir, f) for f in os.listdir(self.storage_dir) if f.endswith('.mp4')],
                        key=os.path.getctime
                    )
                    if mp4_files:
                        oldest_file = mp4_files[0]
                        os.remove(oldest_file)
                        logging.info(f"[PRUNED] Deleted file: {oldest_file}")
            except Exception as e:
                logging.error(f"Error during disk pruning: {e}")
            time.sleep(10)

class RTSPIngestor:
    def __init__(self, camera_id: str, rtsp_url: str, buffer_manager: RingBufferManager, simulate: bool = False):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.buffer_manager = buffer_manager
        self.simulate = simulate
        self.is_running = False

    def start(self):
        self.is_running = True
        thread = threading.Thread(target=self._ingest_loop, daemon=True)
        thread.start()

    def _generate_synthetic_frame(self, frame_num: int):
        import numpy as np
        # 640x480 synthetic frame with timestamp text overlay
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        color = (0, 255, 0) if (frame_num // 15) % 2 == 0 else (0, 200, 255)
        cv2.putText(frame, f"CAM: {self.camera_id} | FRAME: {frame_num}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        # Draw simulated bounding box movement
        cx = int(100 + (frame_num * 5) % 400)
        cv2.rectangle(frame, (cx, 200), (cx + 100, 400), color, 2)
        return frame

    def _ingest_loop(self):
        backoff_sec = 1
        frame_count = 0
        while self.is_running:
            if self.simulate:
                logging.info(f"[SIMULATED STREAM] Starting synthetic ingestion for Camera: {self.camera_id}")
                writer = None
                start_chunk_time = time.time()
                while self.is_running:
                    frame = self._generate_synthetic_frame(frame_count)
                    frame_count += 1
                    now = time.time()
                    if writer is None or (now - start_chunk_time) >= self.buffer_manager.chunk_duration_sec:
                        if writer:
                            writer.release()
                        chunk_filename = f"{self.camera_id}_{int(now)}.mp4"
                        chunk_path = os.path.join(self.buffer_manager.storage_dir, chunk_filename)
                        h, w, _ = frame.shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(chunk_path, fourcc, 15.0, (w, h))
                        start_chunk_time = now
                    writer.write(frame)
                    time.sleep(1.0 / 15.0) # 15 FPS simulation
                if writer:
                    writer.release()
                break

            logging.info(f"Connecting to RTSP Stream for Camera: {self.camera_id}...")
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                logging.warning(f"[RTSP DISCONNECT] Failed to open stream for {self.camera_id}. Retrying in {backoff_sec}s...")
                time.sleep(backoff_sec)
                backoff_sec = min(backoff_sec * 2, 60) # Exponential backoff capped at 60s
                continue

            logging.info(f"[RTSP SUCCESS] Stream connected for Camera: {self.camera_id}")
            backoff_sec = 1 # Reset backoff on successful connection

            writer = None
            start_chunk_time = time.time()

            while cap.isOpened() and self.is_running:
                ret, frame = cap.read()
                if not ret:
                    logging.error(f"[FRAME READ FAIL] Frame drop or packet loss on camera {self.camera_id}")
                    break

                now = time.time()
                # Create a new 60-second .mp4 file chunk
                if writer is None or (now - start_chunk_time) >= self.buffer_manager.chunk_duration_sec:
                    if writer:
                        writer.release()
                    
                    chunk_filename = f"{self.camera_id}_{int(now)}.mp4"
                    chunk_path = os.path.join(self.buffer_manager.storage_dir, chunk_filename)
                    h, w, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(chunk_path, fourcc, 15.0, (w, h))
                    start_chunk_time = now

                writer.write(frame)

            if writer:
                writer.release()
            cap.release()

if __name__ == "__main__":
    buffer = RingBufferManager(storage_dir="./video_buffer", max_disk_usage_pct=85.0)
    ingestor = RTSPIngestor("CAM_STORE_01", "rtsp://127.0.0.1:8554/live.sdp", buffer, simulate=True)
    ingestor.start()
    logging.info("RTSP Ingestion Engine initialized in simulation mode. Ingesting for 3s...")
    time.sleep(3)
    ingestor.is_running = False

