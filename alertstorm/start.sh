#!/bin/bash

# Start the OpenEnv REST environment server natively in the background
echo "Booting OpenEnv server on port 8000..."
uvicorn server.app:app --host 0.0.0.0 --port 8000 &

# Wait for API to initialize
sleep 3

# Start the Human-In-The-Loop Gradio UI Dashboard on port 7860 (Hugging Face Default)
echo "Booting AlertStorm Gradio Console on port 7860..."
python gradio_app.py
