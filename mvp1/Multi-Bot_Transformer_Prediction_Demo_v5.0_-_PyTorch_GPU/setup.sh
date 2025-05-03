#!/bin/bash

# Check if python3 is available
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 could not be found. Please install Python 3."
    exit 1
fi

# Check if pip is available
if ! python3 -m pip --version &> /dev/null
then
    echo "Error: pip could not be found. Please ensure pip is installed for Python 3."
    exit 1
fi

# Create a virtual environment
echo "Creating virtual environment 'venv'..."
python3 -m venv venv

# Activate the virtual environment
# Note: Activation commands differ between shells (bash/zsh vs fish vs cmd/powershell)
# This script assumes bash/zsh. Adapt if needed.
source venv/bin/activate

# Check if activation was successful (simple check)
if ! command -v python &> /dev/null || ! [[ "$(which python)" == *"venv/bin/python"* ]]; then
 echo "Error: Failed to activate virtual environment."
 exit 1
fi
echo "Virtual environment activated."

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify torch installation and CUDA availability (optional but recommended)
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

echo "Setup complete. To run the server, activate the environment ('source venv/bin/activate') and run './run.sh'"

