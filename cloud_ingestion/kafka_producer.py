"""
Module: Cloud Ingestion & Kafka Producer
Description: High-Throughput gRPC Event Ingest & Kafka Producer with Local Offline Buffer Queue.
Author: Prince Sahu
"""

import json
import time
import queue
import logging
import threading
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class CloudKafkaProducer:
    """
    Simulates high-throughput Kafka Event Producer for edge-to-cloud telemetry and event metadata.
    Includes offline queue buffering during WAN drops to ensure zero data loss.
    """
    def __init__(self, bootstrap_servers: str = "localhost:9092", topic_events: str = "camera.events.raw", topic_heartbeats: str = "camera.heartbeat"):
        self.bootstrap_servers = bootstrap_servers
        self.topic_events = topic_events
        self.topic_heartbeats = topic_heartbeats
        self.offline_queue = queue.Queue(maxsize=10000)
        self.is_connected = True
        self.sent_events_count = 0
        self.buffered_events_count = 0

        # Background flush worker thread
        self._running = True
        self._worker = threading.Thread(target=self._flush_buffer_worker, daemon=True)
        self._worker.start()

    def send_event(self, event_data: Dict[str, Any]) -> bool:
        """
        Publishes an edge AI event metadata payload to the Kafka event topic.
        If network/broker drops, buffers to NVMe/memory queue.
        """
        payload = {
            'timestamp': event_data.get('timestamp', time.time()),
            'camera_id': event_data.get('camera_id', 'UNKNOWN_CAM'),
            'track_id': event_data.get('track_id', 'N/A'),
            'event_type': event_data.get('event_type', 'GENERAL_ALERT'),
            'confidence': event_data.get('confidence', 0.0),
            'spatial_validated': event_data.get('spatial_validated', True),
            'persistence_sec': event_data.get('persistence_sec', 3.0),
            'schema_version': '1.0'
        }

        if self.is_connected:
            # Simulate high-throughput Kafka message publish
            logging.info(f"[KAFKA PUBLISH] Topic: {self.topic_events} | Cam: {payload['camera_id']} | Type: {payload['event_type']}")
            self.sent_events_count += 1
            return True
        else:
            # Buffer offline
            try:
                self.offline_queue.put_nowait(payload)
                self.buffered_events_count += 1
                logging.warning(f"[WAN DROP - BUFFERED] Event stored in local queue. Queue Size: {self.offline_queue.qsize()}")
                return False
            except queue.Full:
                logging.error("[BUFFER CRITICAL] Local queue full! Dropping event.")
                return False

    def send_heartbeat(self, camera_id: str, status: str = "ONLINE", disk_usage_pct: float = 45.0):
        """
        Publishes camera edge gateway heartbeat status every 10 seconds.
        """
        heartbeat_payload = {
            'timestamp': time.time(),
            'camera_id': camera_id,
            'status': status,
            'disk_usage_pct': disk_usage_pct
        }
        logging.info(f"[KAFKA HEARTBEAT] Topic: {self.topic_heartbeats} | Cam: {camera_id} | Status: {status} | Disk: {disk_usage_pct:.1f}%")

    def simulate_wan_drop(self, drop: bool = True):
        self.is_connected = not drop
        state = "DISCONNECTED" if drop else "RECONNECTED"
        logging.warning(f"[NETWORK SIMULATOR] Cloud gRPC/Kafka Connection: {state}")

    def _flush_buffer_worker(self):
        while self._running:
            if self.is_connected and not self.offline_queue.empty():
                try:
                    payload = self.offline_queue.get_nowait()
                    logging.info(f"[BACKFILL FLUSH] Re-sending buffered event: Cam {payload['camera_id']} | Event {payload['event_type']}")
                    self.sent_events_count += 1
                    self.offline_queue.task_done()
                except queue.Empty:
                    pass
            time.sleep(0.5)

    def close(self):
        self._running = False

if __name__ == "__main__":
    producer = CloudKafkaProducer()
    logging.info("Kafka Producer Module initialized.")
    
    # Send test event
    test_event = {
        'camera_id': 'CAM_STORE_01',
        'track_id': 'T_99',
        'event_type': 'UNAUTHORIZED_INTRUSION',
        'confidence': 0.92
    }
    producer.send_event(test_event)
    producer.send_heartbeat('CAM_STORE_01')

    # Test WAN drop buffering
    producer.simulate_wan_drop(True)
    producer.send_event(test_event)
    time.sleep(1)

    # Test backfill recovery
    producer.simulate_wan_drop(False)
    time.sleep(1.5)
    producer.close()
