import random
import secrets
from uuid import uuid4
from typing import Dict, List, Any
from collections import deque

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import AlertstormAction, AlertstormObservation, AlertstormState
except (ImportError, ModuleNotFoundError):
    from models import AlertstormAction, AlertstormObservation, AlertstormState

DEPENDENCY_GRAPH = {
    "API_Gateway": ["Auth_Service", "Order_Service"],
    "Auth_Service": ["User_DB", "Redis_Cache"],
    "Order_Service": ["Inventory_DB", "Stripe_API", "Notification_Service"],
    "Notification_Service": [],
    "User_DB": [],
    "Inventory_DB": [],
    "Stripe_API": [],
    "Redis_Cache": []
}

ENTERPRISE_DEPENDENCY_GRAPH = {
    "API_Gateway": ["Auth_Service", "Order_Service", "CDN_Edge", "GraphQL_Server"],
    "Auth_Service": ["User_DB_Primary", "User_Cache_Cluster", "OAuth_Provider_External"],
    "Order_Service": ["Inventory_Service", "Payment_Gateway", "Message_Queue"],
    "GraphQL_Server": ["Product_Catalog", "Recommendation_Engine"],
    "CDN_Edge": ["Static_Assets_S3"],
    "User_DB_Primary": ["EBS_Volume_A", "RDS_Proxy"],
    "User_Cache_Cluster": ["Redis_Node_1", "Redis_Node_2"],
    "Inventory_Service": ["Inventory_DB", "ElasticSearch_Index"],
    "Payment_Gateway": ["Stripe_API", "Fraud_Detection_AI"],
    "Message_Queue": ["Kafka_Broker_Leader", "Zookeeper_Ensemble"],
    "Product_Catalog": ["MongoDB_Cluster", "Image_Optimizer"],
    "Recommendation_Engine": ["Spark_Job_Runner"],
    "Static_Assets_S3": [],
    "EBS_Volume_A": [],
    "RDS_Proxy": [],
    "Redis_Node_1": [],
    "Redis_Node_2": [],
    "Inventory_DB": [],
    "ElasticSearch_Index": [],
    "Stripe_API": [],
    "Fraud_Detection_AI": [],
    "Kafka_Broker_Leader": [],
    "Zookeeper_Ensemble": [],
    "MongoDB_Cluster": [],
    "Image_Optimizer": [],
    "Spark_Job_Runner": [],
    "OAuth_Provider_External": []
}

# ===========================================================================
# MODULE-LEVEL SINGLETON STATE
# The OpenEnv framework creates a new AlertstormEnvironment instance for each
# HTTP request, so instance variables are lost between calls. We use global
# module-level variables to persist state across the /reset -> /step lifecycle.
# ===========================================================================
TASK_OVERRIDE = None

_GLOBAL_STATE = {
    "episode_id": str(uuid4()),
    "step_count": 0,
    "task_level": "standard_easy",
    "secret_root_causes": [],
    "active_alerts": [],
    "suppressed_alerts": [],
    "rewarded_nodes": [],
    "rewarded_suppressions": [],
    "time_elapsed": 0,
    "graph": DEPENDENCY_GRAPH,
}


class AlertstormEnvironment(Environment):
    """
    AlertStorm Environment: 8-Node & 29-Node DAG for Microservice Root Cause Analysis.
    Uses module-level state to survive the OpenEnv framework's per-request instantiation.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        pass  # State lives in _GLOBAL_STATE, not on self

    def _generate_alert(self, service: str, alert_type: str = "Latency Spike", is_noise: bool = False) -> Dict[str, Any]:
        return {
            "id": f"alert-{uuid4().hex[:8]}",
            "service": service,
            "type": alert_type,
            "timestamp": _GLOBAL_STATE["time_elapsed"],
            "is_noise": is_noise
        }

    def _get_realistic_log(self, target: str, is_root_cause: bool, is_noise: bool) -> str:
        if is_root_cause:
            fails = [
                f"CrashLoopBackOff: Pod {target}-7f8b9 crashes with Exit Code 137 (OOMKilled).",
                f"DataIntegrityFault: Deadlock detected in {target} transaction pipeline.",
                f"FATAL: Missing table schema in {target} cluster."
            ]
            return secrets.choice(fails)
        if is_noise:
            noises = [
                f"WARN [Flapping Metric]: {target} CPU spike detected but resolved within 500ms.",
                f"Ghost Metric: {target} Queue backed up temporarily. Proceeding normally."
            ]
            return secrets.choice(noises)
        return f"TimeoutError: Connection to {target} (10.4.2.XX) dropped after 30000ms due to upstream failure."

    def reset(self) -> AlertstormObservation:
        global TASK_OVERRIDE, _GLOBAL_STATE

        if TASK_OVERRIDE:
            task_choice = TASK_OVERRIDE
        else:
            task_choice = secrets.choice(
                ["standard_easy", "standard_medium", "standard_hard", "enterprise_easy", "enterprise_medium", "enterprise_hard"]
            )

        is_enterprise = task_choice.startswith("enterprise_")
        graph = ENTERPRISE_DEPENDENCY_GRAPH if is_enterprise else DEPENDENCY_GRAPH

        # Reset module-level state
        _GLOBAL_STATE["episode_id"] = str(uuid4())
        _GLOBAL_STATE["step_count"] = 0
        _GLOBAL_STATE["task_level"] = task_choice
        _GLOBAL_STATE["time_elapsed"] = 0
        _GLOBAL_STATE["active_alerts"] = []
        _GLOBAL_STATE["suppressed_alerts"] = []
        _GLOBAL_STATE["rewarded_nodes"] = []
        _GLOBAL_STATE["rewarded_suppressions"] = []
        _GLOBAL_STATE["graph"] = graph

        leaves = [n for n, deps in graph.items() if not deps]
        num_rc = 2 if "hard" in task_choice else 1
        _GLOBAL_STATE["secret_root_causes"] = random.sample(leaves, min(num_rc, len(leaves)))

        # Cascade upwards via reverse graph
        reverse_graph = {n: [] for n in graph}
        for parent, children in graph.items():
            for child in children:
                reverse_graph[child].append(parent)

        failing_nodes = set(_GLOBAL_STATE["secret_root_causes"])
        q = deque(_GLOBAL_STATE["secret_root_causes"])
        while q:
            curr = q.popleft()
            for parent in reverse_graph.get(curr, []):
                if parent not in failing_nodes:
                    failing_nodes.add(parent)
                    q.append(parent)

        # Generate signal alerts
        alerts = []
        for node in failing_nodes:
            if node in _GLOBAL_STATE["secret_root_causes"]:
                alerts.append(self._generate_alert(node, "Response Timeout / 500 Error", is_noise=False))
            else:
                alerts.append(self._generate_alert(node, "Response Timeout / 500 Error", is_noise=False))

        # Generate noise / ghost metrics
        noise_budget = 0
        if "medium" in task_choice:
            noise_budget = 5 if is_enterprise else 2
        elif "hard" in task_choice:
            noise_budget = 15 if is_enterprise else 3

        safe_nodes = [n for n in graph if n not in failing_nodes]
        if noise_budget > 0 and safe_nodes:
            noise_targets = random.sample(safe_nodes, min(noise_budget, len(safe_nodes)))
            for n in noise_targets:
                alerts.append(self._generate_alert(n, "Flapping Metric (Noise)", is_noise=True))

        _GLOBAL_STATE["active_alerts"] = alerts

        return AlertstormObservation(
            active_alerts=_GLOBAL_STATE["active_alerts"],
            dependency_graph=graph,
            recent_logs=f"Environment reset. Task level: {task_choice}. Monitoring tools active.",
            reward=0.0,
            done=False
        )

    def step(self, action: AlertstormAction) -> AlertstormObservation:  # type: ignore[override]
        global _GLOBAL_STATE

        _GLOBAL_STATE["step_count"] += 1
        _GLOBAL_STATE["time_elapsed"] += 1

        recent_logs = None
        reward = 0.0
        done = False

        graph = _GLOBAL_STATE["graph"]
        active_alerts = _GLOBAL_STATE["active_alerts"]
        secret_root_causes = _GLOBAL_STATE["secret_root_causes"]

        if action.action_type == "investigate":
            if not action.targets:
                recent_logs = "No targets provided for investigation."
            else:
                target = action.targets[0]
                if target in secret_root_causes:
                    recent_logs = f"CRITICAL LOGS FOUND: {self._get_realistic_log(target, True, False)}"
                    if target not in _GLOBAL_STATE["rewarded_nodes"]:
                        reward = 0.1
                        _GLOBAL_STATE["rewarded_nodes"].append(target)
                    else:
                        reward = 0.0
                elif any(a['service'] == target and a.get('is_noise') for a in active_alerts):
                    recent_logs = self._get_realistic_log(target, False, True)
                elif target not in graph:
                    recent_logs = f"ERROR: Service '{target}' not found in dependency graph."
                else:
                    recent_logs = self._get_realistic_log(target, False, False)

        elif action.action_type == "suppress_alert":
            if not action.targets:
                recent_logs = "No targets provided for suppression."
            else:
                delta_reward = 0.0
                for target in action.targets:
                    matched_alerts = [
                        a for a in active_alerts
                        if a['service'] == target or a['id'] == target
                    ]

                    if matched_alerts:
                        has_noise = any(a.get('is_noise') for a in matched_alerts)
                        has_signal = any(not a.get('is_noise') for a in matched_alerts)

                        # Reward only first-time useful suppression of true noise.
                        if has_noise and target not in _GLOBAL_STATE["rewarded_suppressions"]:
                            delta_reward += 0.1
                            _GLOBAL_STATE["rewarded_suppressions"].append(target)

                        # Penalize suppressing real cascading/signal alerts.
                        if has_signal:
                            delta_reward -= 0.05

                    _GLOBAL_STATE["suppressed_alerts"].append(target)
                    _GLOBAL_STATE["active_alerts"] = [
                        a for a in active_alerts
                        if a['service'] != target and a['id'] != target
                    ]
                active_alerts = _GLOBAL_STATE["active_alerts"]
                recent_logs = f"Muted alerts for {action.targets}."
                reward = round(delta_reward, 2)

        elif action.action_type == "propose_root_cause":
            done = True
            proposed = set(action.targets)
            actual = set(secret_root_causes)

            if proposed == actual:
                reward = 1.0
                recent_logs = f"SUCCESS: Root cause isolated correctly: {list(actual)}. Incident resolved."
            else:
                reward = 0.0
                recent_logs = f"FAILURE: Proposed {list(proposed)}, but true root cause was {list(actual)}."

        return AlertstormObservation(
            active_alerts=active_alerts,
            dependency_graph=graph,
            recent_logs=recent_logs,
            reward=reward,
            done=done
        )

    @property
    def state(self) -> AlertstormState:
        return AlertstormState(
            episode_id=_GLOBAL_STATE["episode_id"],
            step_count=_GLOBAL_STATE["step_count"],
            task_level=_GLOBAL_STATE["task_level"],
            secret_root_causes=_GLOBAL_STATE["secret_root_causes"],
            active_alerts=_GLOBAL_STATE["active_alerts"],
            suppressed_alerts=_GLOBAL_STATE["suppressed_alerts"],
            rewarded_nodes=_GLOBAL_STATE["rewarded_nodes"],
            time_elapsed=_GLOBAL_STATE["time_elapsed"],
        )
