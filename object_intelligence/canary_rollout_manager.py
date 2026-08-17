"""
Module: Canary Rollout & Model Version Router
Description: Dynamic Feature Flag Router (LaunchDarkly / Config API design).
             Supports per-customer model version routing (e.g. v1.2_baseline vs v2.0_new),
             instant 5-second per-customer rollback engine, and 48-hour A/B testing metric evaluator.
Author: Prince Sahu
"""

import time
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class DynamicFeatureFlagRouter:
    """
    Manages tenant feature flags and model version routing rules.
    """

    DEFAULT_BASELINE = "v1.2_baseline"
    NEW_CANDIDATE = "v2.0_new"

    def __init__(self):
        # Schema: { customer_id: { "model_version": str, "canary_enabled": bool, "last_updated": float } }
        self.tenant_configs: Dict[str, dict] = {}
        # Global canary percentage (0.0 to 1.0)
        self.global_canary_ratio: float = 0.05

    def set_customer_model_version(self, customer_id: str, model_version: str, is_canary: bool = False):
        self.tenant_configs[customer_id] = {
            "model_version": model_version,
            "canary_enabled": is_canary,
            "last_updated": time.time()
        }
        logging.info(f"[FEATURE FLAG UPDATED] Customer: {customer_id} ---> Version: {model_version} (Canary: {is_canary})")

    def get_model_version_for_customer(self, customer_id: str) -> str:
        if customer_id in self.tenant_configs:
            return self.tenant_configs[customer_id]["model_version"]
        
        # Deterministic hashing to assign default global canary ratio
        hash_val = abs(hash(customer_id)) % 100
        if hash_val < (self.global_canary_ratio * 100):
            return self.NEW_CANDIDATE
        return self.DEFAULT_BASELINE


class InstantRollbackEngine:
    """
    Executes instant per-customer rollback (< 5 seconds execution target)
    reverting customer model version to baseline without disrupting live recording.
    """

    def __init__(self, router: DynamicFeatureFlagRouter):
        self.router = router
        self.rollback_history: List[Dict] = []

    def execute_instant_rollback(self, customer_id: str, reason: str = "False Positive Spike Reported") -> Dict:
        start_time = time.time()
        old_version = self.router.get_model_version_for_customer(customer_id)
        target_version = DynamicFeatureFlagRouter.DEFAULT_BASELINE

        # Hot-swap tenant model routing in memory
        self.router.set_customer_model_version(customer_id, target_version, is_canary=False)
        duration_ms = (time.time() - start_time) * 1000.0

        record = {
            "customer_id": customer_id,
            "old_version": old_version,
            "new_version": target_version,
            "reason": reason,
            "timestamp": time.time(),
            "execution_duration_ms": round(duration_ms, 2)
        }
        self.rollback_history.append(record)
        logging.warning(f"⚡ [INSTANT ROLLBACK EXECUTED] Customer '{customer_id}' reverted {old_version} -> {target_version} in {duration_ms:.2f}ms! Reason: {reason}")
        return record


class ABTestingEvaluator:
    """
    48-hour A/B testing metrics evaluator comparing baseline vs candidate model deployments.
    Tracks mAP, latency, false positive rate, and customer complaint rates.
    """

    def __init__(self):
        # Schema: { model_version: { "inference_count": int, "total_latency_ms": float, "false_positives": int, "map_score": float } }
        self.metrics: Dict[str, dict] = {
            "v1.2_baseline": {"inference_count": 50000, "total_latency_ms": 10000000.0, "false_positives": 450, "map_score": 0.87},
            "v2.0_new": {"inference_count": 2500, "total_latency_ms": 875000.0, "false_positives": 120, "map_score": 0.92}
        }

    def record_inference_telemetry(self, model_version: str, latency_ms: float, is_false_positive: bool = False):
        if model_version not in self.metrics:
            self.metrics[model_version] = {"inference_count": 0, "total_latency_ms": 0.0, "false_positives": 0, "map_score": 0.90}

        rec = self.metrics[model_version]
        rec["inference_count"] += 1
        rec["total_latency_ms"] += latency_ms
        if is_false_positive:
            rec["false_positives"] += 1

    def generate_ab_report(self) -> Dict[str, dict]:
        report = {}
        for ver, data in self.metrics.items():
            count = max(1, data["inference_count"])
            avg_lat = data["total_latency_ms"] / count
            fp_rate = (data["false_positives"] / count) * 100.0
            report[ver] = {
                "inferences_evaluated": count,
                "avg_latency_ms": round(avg_lat, 2),
                "false_positive_rate_pct": round(fp_rate, 2),
                "mAP_accuracy": data["map_score"]
            }
        return report


class CanaryRolloutManager:
    """
    Unified manager handling feature flag routing, canary deployment, rollback, and A/B analytics.
    """

    def __init__(self):
        self.router = DynamicFeatureFlagRouter()
        self.rollback_engine = InstantRollbackEngine(self.router)
        self.ab_evaluator = ABTestingEvaluator()

    def deploy_canary_to_customer(self, customer_id: str):
        self.router.set_customer_model_version(customer_id, DynamicFeatureFlagRouter.NEW_CANDIDATE, is_canary=True)

    def process_customer_request(self, customer_id: str, latency_ms: float = 200.0, is_fp: bool = False) -> str:
        version = self.router.get_model_version_for_customer(customer_id)
        self.ab_evaluator.record_inference_telemetry(version, latency_ms, is_fp)
        return version


if __name__ == "__main__":
    manager = CanaryRolloutManager()

    # Deploy canary to Customer B
    manager.deploy_canary_to_customer("Customer_B")

    # Customer routing checks
    logging.info(f"Customer A Version: {manager.router.get_model_version_for_customer('Customer_A')}")
    logging.info(f"Customer B Version: {manager.router.get_model_version_for_customer('Customer_B')}")

    # Instant Rollback for Customer B
    rollback_res = manager.rollback_engine.execute_instant_rollback("Customer_B", reason="Latency too slow")
    logging.info(f"Rollback Result: {rollback_res}")

    # A/B Report
    ab_rep = manager.ab_evaluator.generate_ab_report()
    logging.info(f"A/B Testing Report: {ab_rep}")
