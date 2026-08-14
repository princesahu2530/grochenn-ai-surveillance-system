# Enterprise AI CCTV & Surveillance System Architecture
## Production-Grade Distributed Video Ingestion & AI Analytics Platform (100 to 10,000+ Cameras)

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)
![Kafka](https://img.shields.io/badge/Apache-Kafka-black.svg)
![TensorRT](https://img.shields.io/badge/NVIDIA-TensorRT-76B900.svg)

---

## 📌 Executive Summary

This repository contains the full production architecture, edge ingestion engine, AI pipeline code, false alarm reduction cascade, and observability stack for scaling real-time CCTV surveillance systems from 100 to 10,000+ cameras.

### Key Architectural Pillars:
1. **Edge-First Durability, Cloud-Centric Intelligence**: Local Edge Gateways (e.g., NVIDIA Jetson Orin Nano) handle 24/7 RTSP ingestion, NVMe disk-backed circular ring buffering, and 5-FPS detection/tracking. Only AI-validated event metadata and 10-second H.265 event clips are uploaded to the cloud.
2. **84.6% Cost Reduction**: Cuts monthly per-camera bandwidth & cloud compute cost from **$41.45/camera** down to **$6.38/camera/month** for a 5,000-camera fleet.
3. **Zero-Data-Loss Network Failover**: Autonomous edge buffering on NVMe SSDs during WAN drops, with automatic rate-limited background backfill upon network recovery.
4. **Multi-Stage False Alarm Cascade**: Reduces false positive alerts by >85% using spatial polygon masking, temporal persistence debouncing, event-specific dynamic thresholding, and active user feedback loops.

---

## 🏗️ System Architecture Topology

```
+-----------------------------------------------------------------------------------+
|                                  EDGE LOCATION (Store/Site)                       |
|                                                                                   |
|  +--------------------+        RTSP / H.265      +-----------------------------+  |
|  |  CCTV Cameras      | ------------------------>| Edge Gateway Box            |  |
|  | (10 - 50 / location)|                          | (Jetson Orin / Industrial PC)|  |
|  +--------------------+                          +--------------+--------------+  |
|                                                                 |                 |
|                                         +-----------------------+-----------------+  |
|                                         |                       |                 |
|                                         v                       v                 |
|                                 +---------------+       +---------------+         |
|                                 | Local Disk    |       | Light AI      |         |
|                                 | Ring Buffer   |       | Engine        |         |
|                                 | (NVMe SSD)    |       | (TensorRT)    |         |
|                                 +---------------+       +-------+-------+         |
|                                                                 |                 |
|                                           Filtered Event Clip & | Metadata        |
|                                           Heartbeats (gRPC/TLS) |                 |
+-----------------------------------------------------------------|-----------------+
                                                                  |
                                                                  v
+-----------------------------------------------------------------------------------+
|                                  CLOUD PLATFORM (AWS / GCP / Bare Metal)          |
|                                                                                   |
|                   +-----------------------------------------------+               |
|                   | Load Balancer / Regional Ingestion API (gRPC) |               |
|                   +-----------------------+-----------------------+               |
|                                           |                                       |
|                                           v                                       |
|                   +-----------------------------------------------+               |
|                   | Apache Kafka Event Bus                        |               |
|                   | Topics: camera.heartbeat, camera.events.raw   |               |
|                   +-----------------------+-----------------------+               |
|                                           |                                       |
|           +-------------------------------+-------------------------------+       |
|           |                                                               |       |
|           v                                                               v       |
| +-------------------+                                           +-----------------+
| | Temporal Engine   |                                           | Stream Storage  |
| | (Flink / Faust)   |                                           | Worker (S3/GCS) |
| +---------+---------+                                           +--------+--------+
|           |                                                              |        |
|           v                                                              v        |
| +-------------------+      +--------------------+               +-----------------+
| | False Alarm       | ---->| ScyllaDB/Cassandra |               | Object Storage  |
| | Filter Engine     |      | Metadata & Events  |               | (Video Clips)   |
| +---------+---------+      +--------------------+               +-----------------+
|           |                                                                       |
|           v                                                                       |
| +-------------------+                                                             |
| | Push / Webhook    |                                                             |
| | Alert Router      |                                                             |
| +-------------------+                                                             |
+-----------------------------------------------------------------------------------+
```

---

## 📂 Repository Code Structure

```
├── README.md                           # Master Architecture & System Documentation
├── SYSTEM_DESIGN_ANSWERS.md            # Production AI Evaluation & System Design Answers (Q2.1, Q2.2, Q2.3)
├── requirements.txt                    # Project Dependencies
├── main.py                             # End-to-End CLI Orchestrator & Pipeline Runner
├── dashboard.py                        # Interactive Live Monitoring Web Dashboard Server
├── web/
│   └── index.html                      # Modern Glassmorphic Monitoring Web UI
├── edge_ingestion/
│   └── rtsp_ring_buffer.py             # RTSP Ingestion, Exponential Backoff & Disk Auto-Pruner
├── edge_ai/
│   └── inference_pipeline.py           # Subsampling, YOLOv8 TensorRT, ByteTrack & Debouncer
├── cloud_ingestion/
│   └── kafka_producer.py               # High-Throughput gRPC Event Ingest & Kafka Producer
├── cloud_processing/
│   └── false_alarm_cascade.py          # 4-Layer False Alarm Reduction Cascade Engine
├── observability/
│   └── prometheus_alerts.yml           # Prometheus Monitoring & Alertmanager Rules
└── tests/                              # Automated Unit Test Suite
    ├── test_ring_buffer.py
    ├── test_inference_pipeline.py
    ├── test_kafka_producer.py
    └── test_false_alarm_cascade.py
```

---

## 🧠 Production AI Evaluation & System Design Answers
Detailed, metric-backed evaluation responses for AI pipeline architecture, domain degradation diagnosis, and false alarm reduction at scale:
👉 **[View Full System Design Answers Document](SYSTEM_DESIGN_ANSWERS.md)**

---

## 💡 Quickstart & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
pytest tests/ -v
```

### 3. Run End-to-End CLI Pipeline Orchestrator
```bash
python main.py
```

### 4. Launch Interactive Web Dashboard
```bash
python dashboard.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser to view the real-time live CCTV feed, system stats, NVMe storage usage, and 4-stage false alarm filter logs!

---

## ⚙️ Architecture Code Components

### 1. Edge Ingestion & Disk Ring Buffer (`edge_ingestion/rtsp_ring_buffer.py`)
Handles resilient RTSP decoding, connection failover with exponential backoff (1s-60s), 60-second `.mp4` video chunking, and NVMe SSD disk pruning when storage usage exceeds 85%.

### 2. Multi-Stage Edge AI Pipeline (`edge_ai/inference_pipeline.py`)
Decodes 25 FPS down to 5 FPS, runs YOLOv8-Nano TensorRT detection (~8ms), tracks objects with ByteTrack, evaluates spatial polygon intersections, and applies temporal debouncing.

### 3. Cloud Kafka Event Producer (`cloud_ingestion/kafka_producer.py`)
Streams edge telemetry metadata to Kafka topics (`camera.events.raw`, `camera.heartbeat`) with offline NVMe local queue fallback for zero data loss during WAN outages.

### 4. False Alarm Reduction Cascade (`cloud_processing/false_alarm_cascade.py`)
Applies spatial polygon validation, temporal duration persistence checks (min 3s), dynamic confidence score scaling, and business schedule masks to eliminate 85%+ false positive alerts.

---

## 📊 Financial Cost Model Summary (5,000 Cameras)

| Cost Component | Baseline Cloud Upload | Edge-First Architecture | Savings Strategy |
| :--- | :--- | :--- | :--- |
| **Bandwidth (WAN)** | $77,760 ($15.55/cam) | $7,776 ($1.55/cam) | 90% reduction via Edge H.265 event clips upload |
| **Cloud Storage** | $74,520 ($14.90/cam) | $2,268 ($0.45/cam) | S3 Hot Tier 7 days ──► Glacier Deep Archive |
| **Cloud Compute (GPU)**| $45,000 ($9.00/cam) | $9,000 ($1.80/cam) | 90% offloaded to local Edge Jetson nodes |
| **Edge Hardware Amortization**| $0.00 | $6,900 ($1.38/cam) | Industrial Edge Box ($499 per 10 cameras) |
| **Ops & Maintenance** | $10,000 ($2.00/cam) | $6,000 ($1.20/cam) | Centralized K3s fleet & Ansible provisioning |
| **TOTAL MONTHLY** | **$207,280 ($41.45/cam)**| **$31,944 ($6.38/cam)** | **84.6% TOTAL COST REDUCTION** |

---

## 📜 Author
- **Name:** Prince Sahu
- **Email:** princesahu2530@gmail.com

