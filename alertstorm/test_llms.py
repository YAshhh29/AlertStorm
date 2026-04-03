"""
AlertStorm — Multi-LLM Benchmark Runner
========================================
Tests all three tiers of models against the AlertStorm environment.
Keys are loaded from ../.env.alertstorm.local (one level up from alertstorm/).

Tier 1  — Frontier:   Llama 3.3 70B (Groq), Qwen3-32B (Groq), Llama 4 Scout (Groq),
                      GPT-OSS 120B (Groq), Qwen3.6 Plus (OpenRouter), Hermes 405B (OpenRouter),
                      DeepSeek V3 (HuggingFace), Gemini 2.5/3 Pro (Google),
                      Nemotron 3 Super 120B (NVIDIA), MiniMax M2.5 (NVIDIA)
Tier 2  — Mid-range:  Llama 3.1 8B (Groq), GPT-OSS 20B (Groq), Nemotron 3 Super (OpenRouter),
                      Qwen3 Next 80B (OpenRouter), Gemma 3 27B (OpenRouter), Gemini 2.5 Flash,
                      Gemini 2.0 Flash (Google), Qwen 2.5 7B (Together/HuggingFace)
Tier 3  — Small:      Llama 3.2 3B (OpenRouter), Nemotron Nano 9B (OpenRouter),
                      LFM 1.2B (OpenRouter), Llama 3 8B (HuggingFace), Gemma 3 4B,
                      Gemini 2.0 Flash Lite (Google)

Usage:
    python test_llms.py                  # run all available providers
    python test_llms.py --tier 1         # only frontier models
    python test_llms.py --tier 3         # only small/weak models
    python test_llms.py --no-enterprise  # skip 29-node tasks (faster)

Changes vs the original file:
  [FIX-1] Tracks already-investigated nodes → passed to LLM so it stops looping
  [FIX-2] Shows alert types in prompt (signal vs noise IS meaningful info)
  [FIX-3] API failures score 0.0 honestly — no random fallback guessing
  [FIX-4] Model IDs verified via live API queries (April 2, 2026)
  [FIX-5] Removed leaf-node hints from system prompt — that was cheating
  [FIX-6] Added Google Gemini, Together AI, and NVIDIA Build providers
"""

import os
import sys
import io
import json
import time
import re
import argparse
import requests
import openai
from pathlib import Path
from datetime import datetime

# Windows terminal fix for emojis
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
ENV_FILE = Path(__file__).resolve().parent.parent / ".env.alertstorm.local"
API_URL  = os.getenv("ALERTSTORM_API_URL", "http://127.0.0.1:8000")


# ── 1. Load API keys ───────────────────────────────────────────────────────────

def load_keys() -> dict:
    known_keys = ["GROQ_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN", "GEMINI_API_KEY", "TOGETHER_API_KEY", "NVIDIA_API_KEY"]
    keys = {k: os.getenv(k, "") for k in known_keys}

    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k in known_keys and v:
                    keys[k] = v
                    os.environ[k] = v
    else:
        print(f"[warn] .env file not found at {ENV_FILE}")
        print("       Set your keys as environment variables instead.")
    return keys


# ── 2. Model Registry ──────────────────────────────────────────────────────────
# [FIX-4] Model IDs verified as active April 2026.
# Last verified: April 2, 2026 via live API queries.
# If you get a 404/400, visit openrouter.ai/models or console.groq.com/docs/models
# and update the "model" field for that entry.

def get_models(keys: dict, tier_filter: int | None = None) -> list[dict]:
    groq     = keys.get("GROQ_API_KEY", "")
    orkey    = keys.get("OPENROUTER_API_KEY", "")
    hf       = keys.get("HF_TOKEN", "")
    gemini   = keys.get("GEMINI_API_KEY", "")
    together = keys.get("TOGETHER_API_KEY", "")
    nvidia   = keys.get("NVIDIA_API_KEY", "")

    # ── TIER 1: Frontier (best reasoning for complex DAGs) ────────────────────
    tier1 = [
        {
            "tier": 1,
            "label": "Llama 3.3 70B Versatile (Groq)",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": groq,
            "model": "llama-3.3-70b-versatile",
            "provider": "groq",
        },
        {
            "tier": 1,
            "label": "Qwen3-32B (Groq)",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": groq,
            "model": "qwen/qwen3-32b",
            "provider": "groq",
        },
        {
            "tier": 1,
            "label": "Llama 4 Scout 17B (Groq)",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": groq,
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "provider": "groq",
        },
        {
            "tier": 1,
            "label": "GPT-OSS 120B (Groq)",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": groq,
            "model": "openai/gpt-oss-120b",
            "provider": "groq",
        },
        {
            "tier": 1,
            "label": "Llama 3.3 70B Instruct Free (OpenRouter)",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": orkey,
            "model": "meta-llama/llama-3.3-70b-instruct:free",
            "provider": "openrouter",
        },
        {
            "tier": 1,
            "label": "Qwen3.6 Plus Free (OpenRouter)",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": orkey,
            "model": "qwen/qwen3.6-plus:free",
            "provider": "openrouter",
        },
        {
            "tier": 1,
            "label": "Hermes 3 Llama 405B Free (OpenRouter)",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": orkey,
            "model": "nousresearch/hermes-3-llama-3.1-405b:free",
            "provider": "openrouter",
        },
        {
            "tier": 1,
            "label": "DeepSeek V3 (HuggingFace Router)",
            "base_url": "https://router.huggingface.co/v1",
            "api_key": hf,
            "model": "deepseek-ai/DeepSeek-V3",
            "provider": "huggingface",
        },
        {
            "tier": 1,
            "label": "Gemini 2.5 Pro (Google)",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": gemini,
            "model": "gemini-2.5-pro",
            "provider": "gemini",
        },
        {
            "tier": 1,
            "label": "Gemini 3 Pro Preview (Google)",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": gemini,
            "model": "gemini-3-pro-preview",
            "provider": "gemini",
        },
        {
            "tier": 1,
            "label": "Nemotron 3 Super 120B (NVIDIA Build)",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key": nvidia,
            "model": "nvidia/nemotron-3-super-120b-a12b",
            "provider": "nvidia",
        },
        {
            "tier": 1,
            "label": "MiniMax M2.5 (NVIDIA Build)",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key": nvidia,
            "model": "minimaxai/minimax-m2.5",
            "provider": "nvidia",
        },
    ]

    # ── TIER 2: Mid-range (fast and capable) ──────────────────────────────────
    tier2 = [
        {
            "tier": 2,
            "label": "Llama 3.1 8B Instant (Groq)",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": groq,
            "model": "llama-3.1-8b-instant",
            "provider": "groq",
        },
        {
            "tier": 2,
            "label": "GPT-OSS 20B (Groq)",
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": groq,
            "model": "openai/gpt-oss-20b",
            "provider": "groq",
        },
        {
            "tier": 2,
            "label": "Nemotron 3 Super 120B Free (OpenRouter)",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": orkey,
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "provider": "openrouter",
        },
        {
            "tier": 2,
            "label": "Qwen3 Next 80B Free (OpenRouter)",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": orkey,
            "model": "qwen/qwen3-next-80b-a3b-instruct:free",
            "provider": "openrouter",
        },
        {
            "tier": 2,
            "label": "Qwen3 Coder Free (OpenRouter)",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": orkey,
            "model": "qwen/qwen3-coder:free",
            "provider": "openrouter",
        },
        {
            "tier": 2,
            "label": "Gemma 3 27B IT Free (OpenRouter)",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": orkey,
            "model": "google/gemma-3-27b-it:free",
            "provider": "openrouter",
        },
        {
            "tier": 2,
            "label": "Qwen 2.5 7B Instruct (HuggingFace)",
            "base_url": "https://router.huggingface.co/v1",
            "api_key": hf,
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "provider": "huggingface",
        },
        {
            "tier": 2,
            "label": "Gemini 2.5 Flash (Google)",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": gemini,
            "model": "gemini-2.5-flash",
            "provider": "gemini",
        },
        {
            "tier": 2,
            "label": "Gemini 2.0 Flash (Google)",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": gemini,
            "model": "gemini-2.0-flash",
            "provider": "gemini",
        },
        {
            "tier": 2,
            "label": "Qwen 2.5 7B Turbo (Together)",
            "base_url": "https://api.together.xyz/v1",
            "api_key": together,
            "model": "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "provider": "together",
        },
    ]

    # ── TIER 3: Small — proves environment discriminates weak models ──────────
    tier3 = [
        {
            "tier": 3,
            "label": "Llama 3.2 3B Instruct Free (OpenRouter)",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": orkey,
            "model": "meta-llama/llama-3.2-3b-instruct:free",
            "provider": "openrouter",
        },
        {
            "tier": 3,
            "label": "Nemotron Nano 9B Free (OpenRouter)",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": orkey,
            "model": "nvidia/nemotron-nano-9b-v2:free",
            "provider": "openrouter",
        },
        {
            "tier": 3,
            "label": "LFM 1.2B Thinking Free (OpenRouter)",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": orkey,
            "model": "liquid/lfm-2.5-1.2b-thinking:free",
            "provider": "openrouter",
        },
        {
            "tier": 3,
            "label": "Llama 3 8B Instruct (HuggingFace)",
            "base_url": "https://router.huggingface.co/v1",
            "api_key": hf,
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "provider": "huggingface",
        },
        {
            "tier": 3,
            "label": "Gemma 3 4B IT (Google)",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": gemini,
            "model": "gemma-3-4b-it",
            "provider": "gemini",
        },
        {
            "tier": 3,
            "label": "Gemini 2.0 Flash Lite (Google)",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": gemini,
            "model": "gemini-2.0-flash-lite",
            "provider": "gemini",
        },
    ]

    all_models = tier1 + tier2 + tier3
    result = []
    for m in all_models:
        if tier_filter and m["tier"] != tier_filter:
            continue
        if not m["api_key"]:
            continue  # skip silently — key not provided
        result.append(m)
    return result


# ── 3. Prompts ─────────────────────────────────────────────────────────────────

# [FIX-5] Removed "leaf nodes are the only root causes" hint — that is cheating.
#         The LLM must reason this out from the dependency graph itself.
# [FIX-2] Alert types ARE shown — signal vs noise is visible in the environment
#         output and is fair game for the LLM to use.
# [FIX-1] "ALREADY INVESTIGATED" section added — this is not cheating, it is
#         just giving the LLM memory of its own past actions (like any real agent).

SYSTEM_PROMPT = """\
You are an expert SRE agent performing Root Cause Analysis on a cascading microservice failure.

ALERT TYPES:
- "Response Timeout / 500 Error" = this service is failing (root cause OR downstream victim)
- "Flapping Metric (Noise)"      = likely a red herring

YOUR TOOLS:
- investigate:        read logs from a service to understand why it is failing
- suppress_alert:     mute noise alerts that are cluttering the picture
- propose_root_cause: when confident, name the service(s) that originally caused the failure

RULES:
- NEVER re-investigate a node listed in "ALREADY INVESTIGATED" — you already have its logs.
- For "hard" / "enterprise_hard" tasks there may be 2 simultaneous root causes — find both before proposing.
- propose_root_cause ends the episode immediately — only call it when you are confident.

Reply with EXACTLY ONE JSON object — no markdown, no explanation, nothing else:
{"action_type": "investigate",        "targets": ["NODE_NAME"]}
{"action_type": "suppress_alert",     "targets": ["NODE_NAME", "NODE_NAME2"]}
{"action_type": "propose_root_cause", "targets": ["NODE_NAME"]}
"""


def build_user_prompt(
    task_level: str,
    active_alerts: list,
    dep_graph: dict,
    recent_logs: str,
    confirmed: list,
    investigated_cleared: list,   # [FIX-1] nodes already checked, not root cause
) -> str:
    is_enterprise = task_level.startswith("enterprise_")

    all_nodes = set(dep_graph.keys())
    for deps in dep_graph.values():
        all_nodes.update(deps)

    # [FIX-2] Split alerts by type — this info is visible in env output, not cheating
    signal_alerts = [
        {"service": a["service"], "type": a.get("type", "unknown")}
        for a in active_alerts
        if "Noise" not in a.get("type", "")
    ]
    noise_alerts = [
        a["service"]
        for a in active_alerts
        if "Noise" in a.get("type", "")
    ]

    return f"""\
Task: {task_level} ({'29-node enterprise' if is_enterprise else '8-node standard'} topology)

VALID NODE NAMES (use only these exact strings in your targets):
{json.dumps(sorted(all_nodes))}

FAILING SERVICE ALERTS (root cause or downstream victim — you must determine which):
{json.dumps(signal_alerts)}

NOISE ALERTS (probably distractors — consider suppressing):
{json.dumps(noise_alerts)}

DEPENDENCY GRAPH (service -> [services it depends on]):
{json.dumps(dep_graph, indent=2)}

LOGS FROM YOUR LAST ACTION:
{recent_logs if recent_logs else 'No action taken yet.'}

CONFIRMED ROOT CAUSES SO FAR (from log evidence):
{json.dumps(confirmed) if confirmed else 'None confirmed yet.'}

ALREADY INVESTIGATED (do NOT call investigate on these again — you have the logs):
{json.dumps(investigated_cleared) if investigated_cleared else 'None yet.'}

What is your next action? Reply with ONE JSON object only."""


# ── 4. Action extraction ───────────────────────────────────────────────────────

def extract_action(content: str, valid_nodes: set) -> dict | None:
    """Try multiple parsing strategies to pull a valid action from LLM output."""
    strategies = [
        # 1. Direct JSON parse
        lambda c: json.loads(c),
        # 2. Strip ```json fences
        lambda c: json.loads(c.split("```json")[1].split("```")[0]) if "```json" in c else None,
        # 3. Strip plain ``` fences
        lambda c: json.loads(c.split("```")[1].split("```")[0]) if "```" in c else None,
        # 4. Find outermost { ... } block
        lambda c: json.loads(c[c.find("{"):c.rfind("}")+1]) if "{" in c else None,
        # 5. Regex: grab any {...} that contains action_type
        lambda c: next(
            (json.loads(m) for m in re.findall(r"\{[^{}]+\}", c.replace("\n", " "))
             if "action_type" in m),
            None,
        ),
    ]
    for fn in strategies:
        try:
            parsed = fn(content)
            if parsed and "action_type" in parsed and "targets" in parsed:
                # Filter hallucinated node names — only keep real ones
                good_targets = [t for t in parsed["targets"] if t in valid_nodes]
                if good_targets:
                    parsed["targets"] = good_targets
                    return parsed
        except Exception:
            pass
    return None


# ── 5. LLM caller ─────────────────────────────────────────────────────────────

def call_llm(
    cfg: dict,
    task_level: str,
    active_alerts: list,
    dep_graph: dict,
    recent_logs: str,
    confirmed: list,
    investigated_cleared: list,   # [FIX-1]
    retries: int = 3,
) -> dict | None:
    """Call the LLM and return a parsed action dict, or None on unrecoverable failure."""
    all_nodes: set = set(dep_graph.keys())
    for deps in dep_graph.values():
        all_nodes.update(deps)

    user_prompt = build_user_prompt(
        task_level, active_alerts, dep_graph,
        recent_logs, confirmed, investigated_cleared,
    )
    client = openai.OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

    for attempt in range(retries):
        try:
            kwargs = dict(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=200,
                timeout=45.0,
            )
            # Try structured JSON mode first — not all providers support it
            try:
                kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
            except Exception:
                del kwargs["response_format"]
                resp = client.chat.completions.create(**kwargs)

            content = resp.choices[0].message.content or ""
            action  = extract_action(content, all_nodes)
            if action:
                return action
            print(f"      [warn] Could not parse action from: {content[:80]!r}")
            return None

        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = 30 if attempt == 0 else 60
                print(f"      [rate-limit] sleeping {wait}s...")
                time.sleep(wait)
            else:
                # Hard error (404, 400, auth) — no point retrying
                print(f"      [api-error] {err[:120]}")
                return None

    return None


# ── 6. Task evaluator ──────────────────────────────────────────────────────────

TASKS = ["standard_easy", "standard_medium", "standard_hard", "enterprise_easy", "enterprise_medium", "enterprise_hard"]


def _parse_env_response(raw: dict) -> tuple[dict, float, bool]:
    """Unwrap the OpenEnv response envelope."""
    obs    = raw.get("observation", raw)
    reward = raw.get("reward", 0.0)
    done   = raw.get("done", False)
    return obs, reward, done


def evaluate_task(cfg: dict, task_choice: str, verbose: bool = True) -> float:
    """Run one episode. Returns honest score 0.0–1.0 (0.0 on API failure)."""
    max_steps = 25 if task_choice.startswith("enterprise_") else 12

    # Reset environment to the specific task
    try:
        requests.post(f"{API_URL}/reset_with_task", json={"task": task_choice}, timeout=10)
        r = requests.post(f"{API_URL}/reset", timeout=15)
        if r.status_code != 200:
            print(f"      [error] reset failed: {r.status_code}")
            return 0.0
    except Exception as e:
        print(f"      [error] server unreachable: {e}")
        return 0.0

    obs, _, _         = _parse_env_response(r.json())
    active_alerts     = obs.get("active_alerts", [])
    dep_graph         = obs.get("dependency_graph", {})
    recent_logs       = obs.get("recent_logs", "")
    total_reward      = 0.0
    confirmed:            list[str] = []
    investigated_cleared: list[str] = []   # [FIX-1]
    last_investigated:    str | None = None

    for step in range(max_steps):

        # [FIX-1] Parse what the environment told us about the last investigated node
        if last_investigated and recent_logs:
            if "CRITICAL LOGS FOUND" in recent_logs:
                # This node IS a root cause
                if last_investigated not in confirmed:
                    confirmed.append(last_investigated)
            else:
                # This node is NOT a root cause — mark cleared so LLM skips it
                if last_investigated not in investigated_cleared:
                    investigated_cleared.append(last_investigated)

        action = call_llm(
            cfg, task_choice, active_alerts, dep_graph,
            recent_logs, confirmed, investigated_cleared,
        )

        # [FIX-3] API failure → honest 0.0, no random guess
        if action is None:
            print(f"      [skip] API unavailable — scoring as 0.0 (no guess made)")
            return 0.0

        # Track which node was just investigated so we can categorise it next step
        if action.get("action_type") == "investigate" and action.get("targets"):
            last_investigated = action["targets"][0]
        else:
            last_investigated = None

        try:
            raw_step = requests.post(
                f"{API_URL}/step", json={"action": action}, timeout=20
            )
            obs_new, step_reward, done = _parse_env_response(raw_step.json())
            active_alerts = obs_new.get("active_alerts", active_alerts)
            recent_logs   = obs_new.get("recent_logs", "")
            total_reward += step_reward

            if verbose:
                print(
                    f"      [{step+1:2d}] {action['action_type']:22s} "
                    f"{str(action['targets']):30s} → {step_reward:.2f}"
                    + (" ✓ DONE" if done else "")
                )

            if done:
                break

        except Exception as e:
            print(f"      [step-error] {e}")
            break

    return round(min(1.0, max(0.0, total_reward)), 2)


# ── 7. Main runner ─────────────────────────────────────────────────────────────

TIER_LABELS = {1: "[T1] Frontier", 2: "[T2] Mid-Range", 3: "[T3] Small"}


def print_header():
    print()
    print("=" * 80)
    print("  AlertStorm Multi-LLM Benchmark")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  server: {API_URL}")
    print("=" * 80)
    print(f"  Tasks: {', '.join(TASKS)}")
    print("=" * 80)


def print_summary(all_results: dict):
    print("\n\n" + "=" * 80)
    print("  BENCHMARK SUMMARY  (sorted best → worst within each tier)")
    print("  ✅ real  = LLM actually ran   |   ❌ api_err = API failed, score is 0.0")
    print("=" * 80)
    header = (
        f"  {'Model':<44} "
        + "  ".join(f"{t[:7]:>7}" for t in TASKS)
        + f"  {'AVG':>5}  {'STATUS':>9}"
    )
    print(header)
    print("-" * 80)

    last_tier = None
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: (x[1]["tier"], -x[1]["avg"]),
    )

    for label, data in sorted_results:
        tier = data["tier"]
        if tier != last_tier:
            print(f"\n  {TIER_LABELS.get(tier, '')}")
            last_tier = tier
        scores_str = "  ".join(f"{data['scores'].get(t, 0.0):>7.2f}" for t in TASKS)
        status = "✅ real" if data.get("api_ok") else "❌ api_err"
        print(f"  {label:<44} {scores_str}  {data['avg']:>5.2f}  {status:>9}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="AlertStorm Multi-LLM Benchmark")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=None,
                        help="Only test a specific tier (1=frontier, 2=mid, 3=small)")
    parser.add_argument("--tasks", nargs="+", choices=TASKS + ["all"], default=["all"],
                        help="Which tasks to run (default: all)")
    parser.add_argument("--no-enterprise", action="store_true",
                        help="Skip enterprise_* tasks (faster run)")
    args = parser.parse_args()

    task_list = TASKS
    if "all" not in args.tasks:
        task_list = args.tasks
    if args.no_enterprise:
        task_list = [t for t in task_list if not t.startswith("enterprise_")]

    keys   = load_keys()
    models = get_models(keys, tier_filter=args.tier)

    if not models:
        print("\n❌ No models available — check your API keys in .env.alertstorm.local")
        print("   Keys needed: GROQ_API_KEY, OPENROUTER_API_KEY, HF_TOKEN")
        sys.exit(1)

    # Verify server is alive before starting
    try:
        requests.get(f"{API_URL}/tasks", timeout=5)
    except Exception:
        print(f"\n❌ AlertStorm server not reachable at {API_URL}")
        print("   Start it first:")
        print("   cd alertstorm && uvicorn server.app:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    print_header()
    print(f"\n  Models to test: {len(models)}")
    for m in models:
        avail = "✓" if m["api_key"] else "✗ (no key)"
        print(f"    [{TIER_LABELS.get(m['tier'], '?')[0]}] {m['label']}  {avail}")

    all_results: dict = {}

    for cfg in models:
        label = cfg["label"]
        print(f"\n\n{'─'*80}")
        print(f"  🤖  {label}")
        print(f"      Model: {cfg['model']}  |  Provider: {cfg['provider']}")
        print(f"{'─'*80}")

        scores: dict[str, float] = {}
        api_ok = False

        for task in task_list:
            print(f"\n  ▶ Task: {task}")
            score = evaluate_task(cfg, task, verbose=True)
            scores[task] = score
            if score > 0.0:
                api_ok = True
            print(f"    ─── Score: {score:.2f}")
            time.sleep(2)   # gentle rate-limit buffer between tasks

        avg = sum(scores.values()) / max(len(scores), 1)
        all_results[label] = {
            "tier":     cfg["tier"],
            "scores":   scores,
            "avg":      avg,
            "model":    cfg["model"],
            "provider": cfg["provider"],
            "api_ok":   api_ok,
        }
        suffix = "  (⚠️  API errors — scores not meaningful)" if not api_ok else ""
        print(f"\n  ⚡ Average across {len(scores)} tasks: {avg:.2f}{suffix}")

    print_summary(all_results)

    out_path = Path(__file__).parent / "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Full results saved → {out_path.name}")
    print("   Paste the summary table into your README Baseline Scores section.")

if __name__ == "__main__":
    main()
