#!/bin/bash

# Activate the virtual environment (assuming it's in the current directory)
if [ -d "venv" ]; then
  source venv/bin/activate
else
  echo "Error: Virtual environment 'venv' not found. Please run setup.sh first."
  exit 1
fi

# Check if activation was successful
if ! command -v python &> /dev/null || ! [[ "$(which python)" == *"venv/bin/python"* ]]; then
 echo "Error: Failed to activate virtual environment."
 exit 1
fi

# Run the FastAPI server using uvicorn
echo "Starting FastAPI server..."
# Use --reload for development, remove for production
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

