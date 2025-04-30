#!/bin/bash

# Activate virtual environment if it exists and isn't already active
#if [ -d ".venv" ] && [ -z "$VIRTUAL_ENV" ]; then
echo "Activating virtual environment..."
source .venv/bin/activate
#fi

echo "Starting FastAPI server..."
# Use 0.0.0.0 to make it accessible on the network
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
# Remove --reload for production/stable running

echo "Server stopped."


