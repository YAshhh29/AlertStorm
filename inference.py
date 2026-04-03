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
from openai import OpenAI
import os

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# Assuming ALERTSTORM_API_URL is the local environment server you are running
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


# System prompt forces JSON-only output — sent as the system role message
SYSTEM_PROMPT = """You are an SRE automation agent. You must ALWAYS respond with a single valid JSON object and nothing else.
No explanations. No markdown. No prose. No step-by-step reasoning text.
Your entire response must be exactly one of these three formats:
{"action_type": "investigate", "targets": ["NODE_NAME"]}
{"action_type": "suppress_alert", "targets": ["NODE_NAME"]}
{"action_type": "propose_root_cause", "targets": ["NODE_NAME"]}
If you write anything other than a JSON object, you fail the task."""


def get_agent_action(api_key, base_url, model_name,
                     active_alerts, recent_logs, dependency_graph, task_level="standard_easy", history=None):

    is_enterprise = task_level.startswith("enterprise_")
    history_str = "\n".join(history[-5:]) if history else "None yet."

    all_nodes = set(dependency_graph.keys())
    for deps in dependency_graph.values():
        all_nodes.update(deps)
    node_list     = sorted(all_nodes)
    alerted_nodes = [a["service"] for a in active_alerts if "service" in a]

    user_prompt = f"""Task: {task_level} {'(29-node)' if is_enterprise else '(8-node)'}

VALID NODE NAMES (only use these):
{json.dumps(node_list)}

ALERTED NODES:
{json.dumps(active_alerts)}

DEPENDENCY GRAPH (Node depends on [List of Nodes]):
{json.dumps(dependency_graph)}

LOGS:
{recent_logs if recent_logs else 'None yet.'}

PAST ACTIONS HISTORY:
{history_str}

DECISION RULES:
1. Review the currently alerted nodes and logs.
2. Determine which alerts are mere noise and should be suppressed using the `suppress_alert` action.
3. Trace the valid dependencies from upstream back to the origin to find the root cause.
4. If you need more information, you may use the `investigate` action. DO NOT investigate the same node twice if it provided no useful logs.
5. If you are confident you found the root cause, propose it using `propose_root_cause`.
6. Look at PAST ACTIONS HISTORY. Do NOT repeat an action that resulted in a negative reward or did not change the state!

Respond with ONE JSON object only. No text before or after it, no markdown ``` blocks."""

    try:
        if not api_key:
            raise ValueError("No API key provided.")

        client = OpenAI(base_url=base_url, api_key=api_key)

        kwargs = dict(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1000,
            timeout=60.0,
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
        print(f"LLM call failed ({e}), using safe fallback")
        if active_alerts:
            fallback_target = active_alerts[0].get("service", list(dependency_graph.keys())[0])
            return {"action_type": "investigate", "targets": [fallback_target]}
        else:
            return {"action_type": "investigate", "targets": [list(dependency_graph.keys())[0]]}


def _validate(parsed, valid_nodes, active_alerts, dependency_graph, task_level):
    """Reject hallucinated node names before they reach the server."""
    good = [t for t in parsed.get("targets", []) if t in valid_nodes]
    if not good:
        print(f"  [warn] LLM returned invalid targets {parsed.get('targets')} — using fallback")
        if active_alerts:
            fallback_target = active_alerts[0].get("service", list(dependency_graph.keys())[0])
            return {"action_type": "investigate", "targets": [fallback_target]}
        else:
            return {"action_type": "investigate", "targets": [list(dependency_graph.keys())[0]]}
    parsed["targets"] = good
    return parsed


def set_task(task_name):
    try:
        r = requests.post(f"{API_URL}/reset_with_task", json={"task": task_name}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def evaluate_task(api_key, base_url, model_name, task_choice, max_steps=10):
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
    fatal_names = [a.get('service') for a in active_alerts if 'FATAL' in a.get('type', '')]
    print(f"  [debug] alerted={alerted_names} fatal={fatal_names}")

    print("START")
    total_reward = 0.0
    history = []

    for step in range(max_steps):
        action = get_agent_action(
            api_key, base_url, model_name,
            active_alerts, recent_logs, dep_graph, task_choice, history=history
        )

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

            action_str = f"{action.get('action_type')} on {action.get('targets')}"
            history.append(f"Step {step+1}: {action_str} -> reward {step_reward:.2f}")

            print("STEP")
            print(f"Action: {action.get('action_type')} {action.get('targets')}")

            if done:
                break

        except Exception as e:
            continue

    print("END")
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
        scores[task] = evaluate_task(api_key, base_url, model_name, task)
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