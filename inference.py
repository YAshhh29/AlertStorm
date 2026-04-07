"""
AlertStorm Inference Script
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT
- The script emits exactly three line types to stdout:
    [START] task=<task_name> env=alertstorm model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import re
import json
import time
import requests
from typing import List, Optional

from openai import OpenAI

# Environment configuration with defaults
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# AlertStorm environment server
API_URL = os.getenv("ALERTSTORM_API_URL", "http://127.0.0.1:8000")
BENCHMARK = "alertstorm"
MAX_STEPS_STANDARD = 10
MAX_STEPS_ENTERPRISE = 25

def _parse_response(raw: dict) -> tuple:
    obs = raw.get("observation", raw)
    reward = raw.get("reward", 0.0)
    done = raw.get("done", False)
    return obs, reward, done


# ══════════════════════════════════════════════════════════════════════════════
# Logging functions (required stdout format)
# ══════════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Heuristic solver (fallback when LLM unavailable)
# ══════════════════════════════════════════════════════════════════════════════

_solver_state = {
    "investigated": set(),
    "suppressed": set(),
    "discovered": set(),
    "last_investigated": None,
}


def _reset_solver_state():
    _solver_state["investigated"].clear()
    _solver_state["suppressed"].clear()
    _solver_state["discovered"].clear()
    _solver_state["last_investigated"] = None


def heuristic_solver(active_alerts, dependency_graph, task_level, recent_logs=""):
    """Deterministic baseline solver using graph structure."""
    last = _solver_state["last_investigated"]
    if last and recent_logs and "CRITICAL LOGS FOUND" in recent_logs:
        _solver_state["discovered"].add(last)
    
    all_nodes = set(dependency_graph.keys())
    for deps in dependency_graph.values():
        all_nodes.update(deps)
    
    num_needed = 2 if "hard" in task_level else 1
    
    if len(_solver_state["discovered"]) >= num_needed:
        _solver_state["last_investigated"] = None
        return {"action_type": "propose_root_cause", "targets": sorted(_solver_state["discovered"])[:num_needed]}
    
    noise_alerts = [a["service"] for a in active_alerts if "Noise" in a.get("type", "") or "Flapping" in a.get("type", "")]
    unsuppressed = [n for n in noise_alerts if n not in _solver_state["suppressed"]]
    if unsuppressed:
        target = unsuppressed[0]
        _solver_state["suppressed"].add(target)
        _solver_state["last_investigated"] = None
        return {"action_type": "suppress_alert", "targets": [target]}
    
    real_alerts = [a["service"] for a in active_alerts if "Noise" not in a.get("type", "") and "Flapping" not in a.get("type", "")]
    uninvestigated = [n for n in real_alerts if n not in _solver_state["investigated"]]
    uninvestigated.sort(key=lambda n: len(dependency_graph.get(n, [])))
    
    if uninvestigated:
        target = uninvestigated[0]
        _solver_state["investigated"].add(target)
        _solver_state["last_investigated"] = target
        return {"action_type": "investigate", "targets": [target]}
    
    if real_alerts:
        candidates = sorted(real_alerts, key=lambda n: len(dependency_graph.get(n, [])))
        _solver_state["last_investigated"] = None
        return {"action_type": "propose_root_cause", "targets": candidates[:num_needed]}
    
    _solver_state["last_investigated"] = "API_Gateway"
    return {"action_type": "investigate", "targets": ["API_Gateway"]}


# ══════════════════════════════════════════════════════════════════════════════
# LLM-based agent (uses OpenAI client)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an SRE automation agent. Respond with a single valid JSON object only.
Format: {"action_type": "investigate"|"suppress_alert"|"propose_root_cause", "targets": ["NODE_NAME"]}
No explanations, no markdown, just JSON."""


def get_llm_action(client, active_alerts, recent_logs, dependency_graph, task_level, history):
    all_nodes = set(dependency_graph.keys())
    for deps in dependency_graph.values():
        all_nodes.update(deps)
    node_list = sorted(all_nodes)
    
    user_prompt = f"""Task: {task_level}
Nodes: {json.dumps(node_list)}
Alerts: {json.dumps(active_alerts)}
Graph: {json.dumps(dependency_graph)}
Logs: {recent_logs or 'None'}
History: {chr(10).join(history[-5:]) if history else 'None'}
Respond with JSON only."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
            timeout=30.0,
        )
        content = response.choices[0].message.content
        
        for extractor in [
            lambda c: json.loads(c),
            lambda c: json.loads(c[c.find("{"):c.rfind("}")+1]) if "{" in c else None,
        ]:
            try:
                parsed = extractor(content)
                if parsed and "action_type" in parsed and "targets" in parsed:
                    good_targets = [t for t in parsed["targets"] if t in node_list]
                    if good_targets:
                        parsed["targets"] = good_targets
                        return parsed
            except:
                pass
        raise ValueError("Invalid JSON response")
    except Exception:
        return None


def set_task(task_name):
    try:
        r = requests.post(f"{API_URL}/reset_with_task", json={"task": task_name}, timeout=10)
        return r.status_code == 200
    except:
        return False


def evaluate_task(task_name: str, client=None) -> float:
    _reset_solver_state()
    set_task(task_name)
    
    max_steps = MAX_STEPS_ENTERPRISE if task_name.startswith("enterprise_") else MAX_STEPS_STANDARD
    rewards: List[float] = []
    steps_taken = 0
    success = False
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        raw = requests.post(f"{API_URL}/reset", timeout=15)
        if raw.status_code != 200:
            log_end(success=False, steps=0, score=0.01, rewards=[])
            return 0.01
        
        obs, _, _ = _parse_response(raw.json())
        active_alerts = obs.get("active_alerts", [])
        dep_graph = obs.get("dependency_graph", {})
        recent_logs = obs.get("recent_logs", "")
        history = []
        
        for step in range(1, max_steps + 1):
            action = None
            if client:
                action = get_llm_action(client, active_alerts, recent_logs, dep_graph, task_name, history)
            if action is None:
                action = heuristic_solver(active_alerts, dep_graph, task_name, recent_logs)
            
            action_str = f"{action['action_type']}({','.join(action['targets'])})"
            
            try:
                raw_step = requests.post(f"{API_URL}/step", json={"action": action}, timeout=15)
                if raw_step.status_code != 200:
                    log_step(step=step, action=action_str, reward=0.0, done=False, error=f"HTTP {raw_step.status_code}")
                    rewards.append(0.0)
                    steps_taken = step
                    continue
                
                obs, step_reward, done = _parse_response(raw_step.json())
                active_alerts = obs.get("active_alerts", [])
                recent_logs = obs.get("recent_logs", "")
                
                rewards.append(step_reward)
                steps_taken = step
                
                log_step(step=step, action=action_str, reward=step_reward, done=done, error=None)
                
                history.append(f"Step {step}: {action_str} -> {step_reward:.2f}")
                
                if done:
                    success = step_reward >= 1.0
                    break
                    
            except Exception as e:
                log_step(step=step, action=action_str, reward=0.0, done=False, error=str(e))
                rewards.append(0.0)
                steps_taken = step
                continue
        
        score = rewards[-1] if rewards else 0.01
        score = min(max(score, 0.01), 0.99)  # Strictly between 0 and 1
        
    except Exception:
        score = 0.01  # Strictly > 0
        success = False
    
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


def run_baseline():
    client = None
    if API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except:
            pass
    
    tasks = [
        "standard_easy", "standard_medium", "standard_hard",
        "enterprise_easy", "enterprise_medium", "enterprise_hard"
    ]
    
    scores = {}
    for task in tasks:
        scores[task] = evaluate_task(task, client)
    
    return scores


if __name__ == "__main__":
    time.sleep(2)
    scores = run_baseline()
    print(json.dumps(scores))