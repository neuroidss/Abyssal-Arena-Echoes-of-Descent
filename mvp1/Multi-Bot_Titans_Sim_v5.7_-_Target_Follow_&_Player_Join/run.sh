#!/bin/bash

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment 'venv' not found. Please run setup.sh first."
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Starting the server (server.py)..."
python server.py

# Deactivate environment when server stops (optional)
# echo "Deactivating virtual environment."
# deactivate
