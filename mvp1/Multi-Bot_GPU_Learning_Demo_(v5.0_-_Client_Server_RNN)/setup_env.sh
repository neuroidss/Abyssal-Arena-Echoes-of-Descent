#!/bin/bash

# Recommended: Use a virtual environment
python3 -m venv .venv
source .venv/bin/activate # On Windows use `.venv\Scripts\activate`

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Environment created in .venv"
echo "To activate, run: source .venv/bin/activate"
echo "To run the server, run: ./run_server.sh"


