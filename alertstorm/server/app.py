# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Alertstorm Environment.

This module creates an HTTP server that exposes the AlertstormEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import AlertstormAction, AlertstormObservation
    from .alertstorm_environment import AlertstormEnvironment
except (ModuleNotFoundError, ImportError):
    from models import AlertstormAction, AlertstormObservation
    from server.alertstorm_environment import AlertstormEnvironment


from fastapi.middleware.cors import CORSMiddleware

# Create the app with web interface and README integration
app = create_app(
    AlertstormEnvironment,
    AlertstormAction,
    AlertstormObservation,
    env_name="alertstorm",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Merge Gradio UI into the main API app
import gradio as gr
try:
    from alertstorm.gradio_app import create_app as create_gradio_app
except ImportError:
    from gradio_app import create_app as create_gradio_app
gradio_app = create_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/")

@app.get("/baseline")
def get_baseline():
    """
    Returns baseline capability.
    Required by OpenEnv standard grader interface.
    """
    import subprocess
    import sys
    import json
    import os
    try:
        env = os.environ.copy()
        repo_root_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "inference.py")
        )
        local_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "inference.py")
        )
        script_path = repo_root_script if os.path.exists(repo_root_script) else local_script

        if not os.path.exists(script_path):
            raise FileNotFoundError("Could not find inference.py in repository root or alertstorm package.")

        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, env=env)

        output = result.stdout
        start = output.find("{")
        end = output.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Baseline script did not emit JSON scores.")
        json_str = output[start:end + 1]
        scores = json.loads(json_str)
        return scores
    except Exception as e:
        return {"baseline_score": 0.0, "status": "failed", "error": str(e)}

@app.get("/tasks")
def get_tasks():
    """
    Returns the list of specific tasks/modes that the environment supports.
    """
    return {
        "tasks": ["standard_easy", "standard_medium", "standard_hard", "enterprise_easy", "enterprise_medium", "enterprise_hard"],
        "description": "standard_easy: Linear Cascade. standard_medium: 3rd Party Lie (ghost metrics). standard_hard: Split Brain. Enterprise variants scale to 29-node DAG."
    }

@app.post("/reset_with_task")
def reset_with_task(payload: dict = None):
    """
    Reset the environment with a specific task.
    This allows deterministic task selection for grading.
    """
    try:
        from server.alertstorm_environment import TASK_OVERRIDE
        import server.alertstorm_environment as env_module
    except ImportError:
        from .alertstorm_environment import TASK_OVERRIDE
        from . import alertstorm_environment as env_module
    
    if payload and "task" in payload:
        env_module.TASK_OVERRIDE = payload["task"]
    
    # The actual reset will be handled by the create_app's /reset endpoint
    # But we need to trigger it here
    return {"status": "task_set", "task": payload.get("task") if payload else None}


@app.post("/grader")
def evaluate_submission(payload: dict):
    """
    Offline/Custom Grading interface.

    Accepts episode metadata and emits a continuous score in [0.0, 1.0]
    using a weighted blend of:
    - correctness (did the agent isolate the true root cause)
    - micro-progress (useful non-terminal reward trajectory)
    - efficiency (fewer steps is better)

    Expected payload fields (all optional):
    - final_reward: float
    - episode_return: float
    - done: bool
    - steps_taken: int
    - max_steps: int
    - provider_failed: bool
    - trace: list[dict] with per-step reward/done/action
    """
    final_reward = float(payload.get("final_reward", 0.0) or 0.0)
    episode_return = float(payload.get("episode_return", final_reward) or 0.0)
    done = bool(payload.get("done", False))
    steps_taken = int(payload.get("steps_taken", 0) or 0)
    max_steps = max(int(payload.get("max_steps", 10) or 10), 1)
    provider_failed = bool(payload.get("provider_failed", False))

    trace = payload.get("trace") or []
    if not isinstance(trace, list):
        trace = []

    # Correct terminal proposal yields final_reward=1.0 in this env.
    correctness = 1.0 if done and final_reward >= 1.0 else 0.0

    # Positive non-terminal rewards represent useful investigation/suppression progress.
    non_terminal_positive = 0.0
    for step in trace:
        if not isinstance(step, dict):
            continue
        reward = float(step.get("reward", 0.0) or 0.0)
        step_done = bool(step.get("done", False))
        if reward > 0.0 and not step_done:
            non_terminal_positive += reward

    # Fallback if trace is unavailable.
    if non_terminal_positive <= 0.0:
        non_terminal_positive = max(0.0, episode_return - (1.0 if correctness else 0.0))

    # Two +0.1 useful micro-actions saturate micro component at 1.0.
    micro_component = min(non_terminal_positive / 0.2, 1.0)

    # Step efficiency in [0, 1], where 1 means solved in first step.
    efficiency = 1.0 - (max(steps_taken - 1, 0) / max(max_steps - 1, 1))
    efficiency = max(0.0, min(1.0, efficiency))

    score = 0.75 * correctness + 0.15 * micro_component + 0.10 * efficiency

    # If provider failed before producing any usable trajectory, keep score at 0.
    if provider_failed and not trace:
        score = 0.0

    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 3),
        "feedback": "Continuous grader evaluated correctness, trajectory, and efficiency.",
        "valid": True,
        "in_range": 0.0 <= score <= 1.0,
        "components": {
            "correctness": round(correctness, 3),
            "micro_progress": round(micro_component, 3),
            "efficiency": round(efficiency, 3),
        },
    }



def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m alertstorm.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)
    """
    import uvicorn
    uvicorn.run("server.app:app", host=host, port=port)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
