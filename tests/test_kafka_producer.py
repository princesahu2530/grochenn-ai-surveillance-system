import time
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloud_ingestion.kafka_producer import CloudKafkaProducer

def test_kafka_producer_online():
    producer = CloudKafkaProducer()
    try:
        event = {
            'camera_id': 'TEST_CAM_01',
            'track_id': 'T_01',
            'event_type': 'LOITERING',
            'confidence': 0.85
        }
        res = producer.send_event(event)
        assert res is True
        assert producer.sent_events_count == 1
    finally:
        producer.close()

def test_kafka_producer_offline_failover():
    producer = CloudKafkaProducer()
    try:
        # Simulate network drop
        producer.simulate_wan_drop(True)
        event = {
            'camera_id': 'TEST_CAM_02',
            'event_type': 'UNAUTHORIZED_INTRUSION',
            'confidence': 0.95
        }
        res = producer.send_event(event)
        assert res is False
        assert producer.offline_queue.qsize() == 1

        # Simulate network recovery
        producer.simulate_wan_drop(False)
        time.sleep(1.0)
        assert producer.offline_queue.qsize() == 0
    finally:
        producer.close()
