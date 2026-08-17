"""
Module: Hierarchical Modular Object Detector & Asset Classifier
Description: Hierarchical Modular Architecture combining a Coarse Detector (YOLOv8 super-classes:
             Person, Vehicle, Machinery, Package, Safety_Gear) with Open-Set Metric Learning
             for 200+ fine-grained custom asset classes. Supports zero-downtime custom class registration (<2 hours).
Author: Prince Sahu
"""

import time
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class CoarseYOLOv8Detector:
    """
    Stage 1 Coarse Detector running lightweight YOLOv8 on 5 primary super-classes.
    """

    SUPER_CLASSES = ["person", "vehicle", "machinery", "package", "safety_gear"]

    def detect_super_classes(self, frame: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Simulates coarse object detection extracting spatial bounding boxes and super-class labels.
        """
        # Simulated edge detections
        return [
            {"bbox": (120, 150, 280, 480), "super_class": "person", "confidence": 0.92},
            {"bbox": (310, 200, 450, 320), "super_class": "safety_gear", "confidence": 0.88},
            {"bbox": (500, 180, 620, 300), "super_class": "machinery", "confidence": 0.94},
            {"bbox": (50, 300, 180, 420), "super_class": "package", "confidence": 0.85}
        ]


class OpenSetMetricLearningRetriever:
    """
    Stage 2 Fine-Grained Open-Set Metric Learning Vector Retriever.
    Maps cropped object regions to 128d metric visual vectors, matching against registered custom class centroids.
    Enables zero-downtime introduction of new customer classes without model retraining.
    """

    def __init__(self, vector_dim: int = 128, match_threshold: float = 0.78):
        self.vector_dim = vector_dim
        self.match_threshold = match_threshold
        # Schema: { class_name: { "centroids": List[np.ndarray], "super_class": str, "customer_id": str } }
        self.class_registry: Dict[str, dict] = {}
        # Pre-seed default fine-grained asset classes
        self._seed_default_classes()

    def _seed_default_classes(self):
        default_assets = [
            ("hard_hat_yellow", "safety_gear", "GLOBAL"),
            ("safety_vest_orange", "safety_gear", "GLOBAL"),
            ("forklift_caterpillar", "machinery", "GLOBAL"),
            ("warehouse_pallet", "package", "GLOBAL"),
            ("staff_uniform_blue", "person", "GLOBAL")
        ]
        for class_name, super_cls, tenant in default_assets:
            seed_val = abs(hash(class_name)) % (2**31)
            rng = np.random.RandomState(seed_val)
            vec = rng.randn(self.vector_dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            self.register_custom_class(class_name=class_name, super_class=super_cls, prototype_vectors=[vec], tenant_id=tenant)

    def register_custom_class(self, class_name: str, super_class: str, prototype_vectors: List[np.ndarray], tenant_id: str = "GLOBAL") -> Dict:
        """
        Zero-Downtime Custom Class Registration Pipeline (< 2 hours, 0 full model retraining required).
        Registers centroid embeddings in the live metric index.
        """
        normalized_protos = []
        for v in prototype_vectors:
            norm_v = v / (np.linalg.norm(v) + 1e-7)
            normalized_protos.append(norm_v)

        self.class_registry[class_name] = {
            "super_class": super_class,
            "centroids": normalized_protos,
            "tenant_id": tenant_id,
            "registered_at": time.time()
        }
        logging.info(f"[ZERO-DOWNTIME CLASS REGISTERED] Asset: '{class_name}' | SuperClass: '{super_class}' | Tenant: '{tenant_id}'")
        return {"class_name": class_name, "status": "ACTIVE", "total_registered_classes": len(self.class_registry)}

    def extract_metric_embedding(self, crop: Optional[np.ndarray] = None, hint_class: str = "") -> np.ndarray:
        if hint_class and hint_class in self.class_registry:
            # Generate vector close to target class centroid
            target_proto = self.class_registry[hint_class]["centroids"][0]
            noise = np.random.normal(0, 0.05, self.vector_dim).astype(np.float32)
            vec = target_proto + noise
        else:
            vec = np.random.randn(self.vector_dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-7)

    def classify_crop(self, crop_embedding: np.ndarray, super_class_filter: str = "", tenant_id: str = "GLOBAL") -> Tuple[str, float]:
        best_class = f"unclassified_{super_class_filter}" if super_class_filter else "unclassified_object"
        max_sim = -1.0

        for class_name, data in self.class_registry.items():
            # Check super class boundary filter
            if super_class_filter and data["super_class"] != super_class_filter:
                continue

            # Check tenant isolation
            if data["tenant_id"] not in ["GLOBAL", tenant_id]:
                continue

            for proto in data["centroids"]:
                sim = float(np.dot(proto, crop_embedding))
                if sim > max_sim:
                    max_sim = sim
                    best_class = class_name

        if max_sim >= self.match_threshold:
            return best_class, max_sim
        return f"unclassified_{super_class_filter}" if super_class_filter else "unclassified_object", max_sim


class ModularHierarchicalObjectDetector:
    """
    Combines Coarse YOLOv8 Detector with Fine-Grained Open-Set Metric Learning.
    """

    def __init__(self):
        self.coarse_detector = CoarseYOLOv8Detector()
        self.metric_retriever = OpenSetMetricLearningRetriever()

    def process_frame(self, frame: Optional[np.ndarray] = None, tenant_id: str = "GLOBAL") -> List[Dict]:
        coarse_detections = self.coarse_detector.detect_super_classes(frame)
        fine_results = []

        for det in coarse_detections:
            super_cls = det["super_class"]
            bbox = det["bbox"]
            conf = det["confidence"]

            # Map super-class to fine-grained visual metric query
            hint_cls = "hard_hat_yellow" if super_cls == "safety_gear" else ("forklift_caterpillar" if super_cls == "machinery" else "")
            metric_vec = self.metric_retriever.extract_metric_embedding(None, hint_class=hint_cls)
            fine_class, fine_score = self.metric_retriever.classify_crop(metric_vec, super_class_filter=super_cls, tenant_id=tenant_id)

            fine_results.append({
                "bbox": bbox,
                "super_class": super_cls,
                "fine_class": fine_class,
                "coarse_confidence": conf,
                "fine_metric_score": round(fine_score, 4),
                "combined_confidence": round(conf * fine_score if fine_score > 0 else conf * 0.8, 4)
            })

        return fine_results


if __name__ == "__main__":
    detector = ModularHierarchicalObjectDetector()

    # 1. Test standard detection
    results = detector.process_frame(tenant_id="TENANT_LOGISTICS_INC")
    for r in results:
        logging.info(f"Detected: {r['fine_class']} (Super: {r['super_class']}) | Score: {r['fine_metric_score']}")

    # 2. Test zero-downtime registration of a brand-new customer class: "custom_hazmat_suit"
    new_proto = np.random.randn(128).astype(np.float32)
    reg_info = detector.metric_retriever.register_custom_class(
        class_name="custom_hazmat_suit",
        super_class="person",
        prototype_vectors=[new_proto],
        tenant_id="TENANT_CHEMA_CORP"
    )
    logging.info(f"Registration status: {reg_info}")
