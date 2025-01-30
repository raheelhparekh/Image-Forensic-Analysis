#!/bin/bash

# Exit on error
set -e

echo "🔄 Running Image Analysis Automation..."

# Ensure dependencies are installed
bash install.sh

# Input parameters
INPUT_IMAGE="$1"
USER="$2"
WEBSITE="$3"
TOOL_NAME="ImageAnalysisTool"

# Run the Python script
if [[ -f "main.py" ]]; then
    if [[ -f "$INPUT_IMAGE" ]]; then
        python3 main.py --original "$INPUT_IMAGE" --user "$USER" --website "$WEBSITE" --tool "$TOOL_NAME"
        echo "✅ Image analysis completed! Results are saved in the Output/ directory."
    else
        echo "❌ Error: Input image not found: $INPUT_IMAGE"
        exit 1
    fi
else
    echo "❌ Error: main.py not found!"
    exit 1
fi
