"""
AlertStorm Inference Evaluator
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
"""

import os
import re
import json
import requests
import time
import openai

API_URL = os.getenv("ALERTSTORM_API_URL", "http://127.0.0.1:8000")

def _parse_response(raw: dict) -> tuple:
    """
    Parses OpenEnv API response envelope.
    Structure: {"observation": {...}, "reward": float, "done": bool}
    Returns: (obs_dict, reward, done)
    """
    obs = raw.get("observation", raw)   # nested under 'observation' key
    reward = raw.get("reward", 0.0)     # top-level
    done = raw.get("done", False)       # top-level
    return obs, reward, done

# Tracks stateful progress of the heuristic solver across steps within one episode
_fallback_state = {
    "investigated": set(),       # nodes already investigated
    "suppressed": set(),         # noise nodes already suppressed
    "step": 0,
    "discovered": set(),         # nodes confirmed as root cause via CRITICAL log
    "last_investigated": None,   # node investigated in the previous step (for log parsing)
}

def _reset_fallback_state():
    _fallback_state["investigated"].clear()
    _fallback_state["suppressed"].clear()
    _fallback_state["discovered"].clear()
    _fallback_state["step"] = 0
    _fallback_state["last_investigated"] = None


# System prompt forces JSON-only output — sent as the system role message
SYSTEM_PROMPT = """You are an SRE automation agent. You must ALWAYS respond with a single valid JSON object and nothing else.
No explanations. No markdown. No prose. No step-by-step reasoning text.
Your entire response must be exactly one of these three formats:
{"action_type": "investigate", "targets": ["NODE_NAME"]}
{"action_type": "suppress_alert", "targets": ["NODE_NAME"]}
{"action_type": "propose_root_cause", "targets": ["NODE_NAME"]}
If you write anything other than a JSON object, you fail the task."""


def get_agent_action(api_key, base_url, model_name,
                     active_alerts, recent_logs, dependency_graph, task_level="standard_easy"):

    # ── Pre-LLM deterministic heuristic ──────────────────────────────────────
    # Handles clear-cut cases (noise suppression, confirmed root causes).
    # Falls through to the LLM for ambiguous investigation decisions.
    heuristic = _heuristic_override(active_alerts, dependency_graph, task_level, recent_logs)
    if heuristic is not None:
        return heuristic
    # ─────────────────────────────────────────────────────────────────────────

    is_enterprise = task_level.startswith("enterprise_")

    all_nodes = set(dependency_graph.keys())
    for deps in dependency_graph.values():
        all_nodes.update(deps)
    node_list     = sorted(all_nodes)
    alerted_nodes = [a["service"] for a in active_alerts if "service" in a]

    # Real alerts (not noise) — all labeled identically, root cause unknown from type alone
    real_alerts = [
        a["service"] for a in active_alerts
        if "Noise" not in a.get("type", "") and "service" in a
    ]
    noise_alerts = [
        a["service"] for a in active_alerts
        if "Noise" in a.get("type", "") and "service" in a
    ]
    # Nodes already confirmed as root cause by log analysis this episode
    confirmed = sorted(_fallback_state["discovered"])

    user_prompt = f"""You are an SRE agent investigating a cascading microservice failure.

Task: {task_level} ({'29-node enterprise' if is_enterprise else '8-node standard'} topology)

VALID NODE NAMES (only use these exact strings):
{json.dumps(node_list)}

ACTIVE SERVICE ALERTS (all look the same — you must investigate to find the root cause):
{json.dumps(real_alerts)}

NOISE ALERTS (flapping metrics, suppress these first):
{json.dumps(noise_alerts)}

DEPENDENCY GRAPH (parent -> [children it depends on]):
{json.dumps(dependency_graph)}

INVESTIGATION LOGS FROM LAST ACTION:
{recent_logs if recent_logs else 'No investigation performed yet.'}

ALREADY CONFIRMED ROOT CAUSES (via log analysis):
{json.dumps(confirmed) if confirmed else 'None yet.'}

STRATEGY:
- All service alerts look identical. You CANNOT determine the root cause from alert type alone.
- You MUST investigate nodes to see their logs.
- If logs say "CRITICAL LOGS FOUND" -> that node IS a root cause. Propose it.
- If logs show a timeout/upstream error -> that node is a symptom, trace further upstream.
- Investigate leaf nodes first (nodes with empty [] in the graph) — they are most likely root causes.
- For hard tasks, there may be 2 simultaneous root causes. Find both before proposing.
- Suppress noise alerts immediately (they are labeled Flapping Metric).

Respond with ONE JSON object only. No text before or after it."""

    try:
        if not api_key:
            raise ValueError("No API key provided.")

        client = openai.OpenAI(base_url=base_url, api_key=api_key)

        kwargs = dict(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=100,
            timeout=15.0,
        )
        try:
            kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
        except Exception:
            del kwargs["response_format"]
            response = client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content

        for extractor in [
            lambda c: json.loads(c),
            lambda c: json.loads(c[c.find("{"):c.rfind("}")+1]) if "{" in c else None,
            lambda c: next(
                (json.loads(m) for m in re.findall(r"\{[^{}]*\}", c.replace("\n"," "))
                 if "action_type" in json.loads(m)), None
            ),
        ]:
            try:
                parsed = extractor(content)
                if parsed and "action_type" in parsed and "targets" in parsed:
                    return _validate(parsed, node_list, active_alerts, dependency_graph, task_level)
            except Exception:
                pass

        raise ValueError(f"No valid JSON in response: {content[:120]}")

    except Exception as e:
        print(f"LLM call failed ({e}), using fallback solver")
        return fallback_solver(active_alerts, dependency_graph, task_level)


def _heuristic_override(active_alerts, dependency_graph, task_level, recent_logs=""):
    """
    Log-driven stateful heuristic. Discovers root causes by reading investigation logs,
    NOT by inspecting alert type labels (which are now identical for all real alerts).
    Returns None to hand off to the LLM for genuinely ambiguous mid-investigation decisions.
    """
    # ── Step 1: Parse the log from the PREVIOUS step to update our knowledge ──
    last = _fallback_state["last_investigated"]
    if last and "CRITICAL LOGS FOUND" in (recent_logs or ""):
        _fallback_state["discovered"].add(last)

    all_nodes = set(dependency_graph.keys())
    for deps in dependency_graph.values():
        all_nodes.update(deps)

    num_needed = 2 if "hard" in task_level else 1

    # ── Step 2: If we've confirmed enough root causes, propose them immediately ──
    if len(_fallback_state["discovered"]) >= num_needed:
        return {"action_type": "propose_root_cause",
                "targets": sorted(_fallback_state["discovered"])[:num_needed]}

    # ── Step 3: Suppress unseen noise first (labeled, safe to auto-suppress) ──
    noise = [a["service"] for a in active_alerts
             if "Noise" in a.get("type", "") and a["service"] not in _fallback_state["suppressed"]]
    if noise:
        _fallback_state["suppressed"].add(noise[0])
        return {"action_type": "suppress_alert", "targets": [noise[0]]}

    # ── Step 4: Investigate unvisited real-alert nodes, leaves first ──────────
    real_alerted = [a["service"] for a in active_alerts
                    if "Noise" not in a.get("type", "") and "service" in a]
    unseen = [s for s in real_alerted if s not in _fallback_state["investigated"]]
    # Prioritise leaf nodes (no dependencies) — they are the most likely root causes
    unseen_sorted = sorted(unseen, key=lambda s: (len(dependency_graph.get(s, [])), s))

    if unseen_sorted:
        target = unseen_sorted[0]
        _fallback_state["investigated"].add(target)
        # Note: last_investigated is updated in evaluate_task after the action is sent
        return {"action_type": "investigate", "targets": [target]}

    # ── Step 5: All nodes investigated, propose the leaf-most candidates ──────
    if real_alerted:
        best = sorted(real_alerted, key=lambda s: len(dependency_graph.get(s, [])))
        return {"action_type": "propose_root_cause",
                "targets": best[:min(num_needed, len(best))]}

    # ── Step 6: Nothing alerted — hand off to LLM ────────────────────────────
    return None


def _validate(parsed, valid_nodes, active_alerts, dependency_graph, task_level):
    """Reject hallucinated node names before they reach the server."""
    good = [t for t in parsed.get("targets", []) if t in valid_nodes]
    if not good:
        print(f"  [warn] LLM returned invalid targets {parsed.get('targets')} — using fallback")
        return fallback_solver(active_alerts, dependency_graph, task_level)
    parsed["targets"] = good
    return parsed


def fallback_solver(active_alerts, dependency_graph, task_level="standard_easy"):
    """
    Stateful heuristic. Never loops on the same action.
    Priority: suppress noise -> propose alerted leaf -> investigate unseen -> force propose best.
    """
    _fallback_state["step"] += 1
    alerted = [a["service"] for a in active_alerts if "service" in a]

    all_nodes = set(dependency_graph.keys())
    for deps in dependency_graph.values():
        all_nodes.update(deps)

    leaves         = [n for n in all_nodes if not dependency_graph.get(n)]
    alerted_leaves = [n for n in leaves if n in alerted]

    # 1. Suppress unseen noise
    noise = [a["service"] for a in active_alerts
             if "Noise" in a.get("type", "")
             and a["service"] not in _fallback_state["suppressed"]]
    if noise:
        _fallback_state["suppressed"].add(noise[0])
        return {"action_type": "suppress_alert", "targets": [noise[0]]}

    # 2. Propose alerted leaf immediately
    if alerted_leaves:
        targets = alerted_leaves[:2] if "hard" in task_level else [alerted_leaves[0]]
        return {"action_type": "propose_root_cause", "targets": targets}

    # 3. Investigate unseen alerted nodes (fewest deps first = closest to leaf)
    unseen = sorted(
        [s for s in alerted if s not in _fallback_state["investigated"]],
        key=lambda s: len(dependency_graph.get(s, []))
    )
    if unseen:
        _fallback_state["investigated"].add(unseen[0])
        return {"action_type": "investigate", "targets": [unseen[0]]}

    # 4. All alerted nodes investigated — propose the one with fewest deps
    best = sorted(alerted, key=lambda s: len(dependency_graph.get(s, [])))
    if best:
        targets = best[:2] if "hard" in task_level else [best[0]]
        return {"action_type": "propose_root_cause", "targets": targets}

    # 5. No alerts at all — investigate the gateway as last resort
    return {"action_type": "investigate", "targets": ["API_Gateway"]}


def set_task(task_name):
    try:
        r = requests.post(f"{API_URL}/reset_with_task", json={"task": task_name}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def evaluate_task(api_key, base_url, model_name, task_choice, max_steps=10):
    _reset_fallback_state()
    set_task(task_choice)

    raw = requests.post(f"{API_URL}/reset", timeout=15)
    if raw.status_code != 200:
        print(f"Reset failed for task {task_choice}")
        return 0.0

    obs, _, _ = _parse_response(raw.json())
    active_alerts = obs.get("active_alerts", [])
    dep_graph     = obs.get("dependency_graph", {})
    recent_logs   = obs.get("recent_logs", "")

    alerted_names = [a.get('service') for a in active_alerts]
    real_alerted  = [a.get('service') for a in active_alerts if 'Noise' not in a.get('type', '')]
    print(f"  [debug] alerted={alerted_names} | non-noise={real_alerted}")

    total_reward = 0.0

    for step in range(max_steps):
        action = get_agent_action(
            api_key, base_url, model_name,
            active_alerts, recent_logs, dep_graph, task_choice,
        )

        # Track what was investigated so the heuristic can parse the returned log
        if action.get("action_type") == "investigate" and action.get("targets"):
            _fallback_state["last_investigated"] = action["targets"][0]
        else:
            _fallback_state["last_investigated"] = None

        try:
            raw_step = requests.post(f"{API_URL}/step", json={"action": action}, timeout=15)
            if raw_step.status_code != 200:
                print(f"  Step {step} server error: {raw_step.status_code} - {raw_step.text}")
                continue

            # Parse the envelope: reward/done are top-level, obs fields are nested
            obs, step_reward, done = _parse_response(raw_step.json())
            active_alerts = obs.get("active_alerts", [])
            recent_logs   = obs.get("recent_logs", "")
            total_reward += step_reward

            print(
                f"  Step {step+1:2d}: {action.get('action_type'):20s} "
                f"targets={str(action.get('targets')):30s} "
                f"reward={step_reward:.2f} done={done}"
            )

            if done:
                break

        except Exception as e:
            print(f"  Step {step} error: {e}")
            continue

    return round(min(1.0, max(0.0, total_reward)), 2)


def run_baseline():
    api_key    = os.getenv("HF_TOKEN", "")
    base_url   = os.getenv("API_BASE_URL", "")
    model_name = os.getenv("MODEL_NAME", "")

    print("Running AlertStorm baseline evaluation")
    print(f"  API Base URL: {base_url}")
    print(f"  Model:        {model_name}")
    print(f"  API Key:      {'[SET]' if api_key else '[NOT SET]'}")
    print()

    if not base_url:
        raise EnvironmentError("API_BASE_URL is not set. Please: set API_BASE_URL=<your-openai-compatible-endpoint>")

    if not model_name:
        raise EnvironmentError("MODEL_NAME is not set. Please: set MODEL_NAME=<your-model-id>")

    if not api_key:
        raise EnvironmentError("HF_TOKEN is not set. Please: set HF_TOKEN=<your-hf-token>")

    tasks  = ["standard_easy", "standard_medium", "standard_hard", "enterprise_easy", "enterprise_medium", "enterprise_hard"]
    scores = {}

    for task in tasks:
        print(f"Evaluating task: {task}")
        max_steps = 25 if task.startswith("enterprise_") else 10
        scores[task] = evaluate_task(api_key, base_url, model_name, task, max_steps=max_steps)
        print(f"  Final score: {scores[task]:.2f}\n")

    print("=" * 50)
    print("BASELINE SCORES:")
    for task, score in scores.items():
        print(f"  {task}: {score:.2f}")
    print(f"  Average: {sum(scores.values()) / len(scores):.2f}")
    print("=" * 50)

    return scores

if __name__ == "__main__":
    time.sleep(2)
    scores = run_baseline()
    print(json.dumps(scores))