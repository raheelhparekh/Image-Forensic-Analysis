#!/bin/bash

# Exit on error
set -e

echo "Running Image Analysis Automation..."

# Ensure dependencies are installed
bash install.sh

# Input parameters
ORIGINAL_IMAGE="$1"
TAMPERED_IMAGE="$2"
USER="$3"
WEBSITE="$4"
TOOL_NAME="ImageAnalysisTool"

# Run the Python script
if [[ -f "main.py" ]]; then
    if [[ -f "$ORIGINAL_IMAGE" && -f "$TAMPERED_IMAGE" ]]; then
        python3 main.py --original "$ORIGINAL_IMAGE" --tampered "$TAMPERED_IMAGE" --user "$USER" --website "$WEBSITE" --tool "$TOOL_NAME"
        echo " Image analysis completed! Results are saved in the Output/ directory."
    else
        echo " Error: Input images not found: $ORIGINAL_IMAGE or $TAMPERED_IMAGE"
        exit 1
    fi
else
    echo " Error: main.py not found!"
    exit 1
fi