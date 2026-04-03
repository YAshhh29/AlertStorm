#!/bin/bash

# Start the unified OpenEnv REST environment server & Gradio UI
echo "Booting AlertStorm Unified Server on port 8000..."
exec uvicorn server.app:app --host 0.0.0.0 --port 8000
