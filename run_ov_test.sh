#!/bin/bash
#
# Script to run OpenVINO Dynamic Speculation Test
# This script activates the virtual environment and runs ov_test.py with any provided arguments
#

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Virtual environment path
VENV_PATH="$SCRIPT_DIR/venv_ov_test"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "❌ Error: Virtual environment not found at $VENV_PATH"
    echo "Please create it first with:"
    echo "  python3 -m venv venv_ov_test"
    echo "  source venv_ov_test/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate virtual environment"
    exit 1
fi

echo "✅ Virtual environment activated"
echo "🚀 Running ov_test.py..."
echo ""

# Run the script with all provided arguments
python "$SCRIPT_DIR/ov_test.py" "$@"

# Capture exit code
EXIT_CODE=$?

# Deactivate virtual environment
deactivate

# Exit with the same code as the Python script
exit $EXIT_CODE
