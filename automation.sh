#!/bin/bash

# Title: Image Analysis Automation Script
# Author: Automation Script
# Description: Automates running the Python script for image analysis without deleting the `outputs` directory.

# Exit on error
set -e

# Paths
PYTHON_SCRIPT="image_analysis.py"  # Replace with the actual Python script filename
ORIGINAL_IMAGE="images/cup_image.jpg"
MODIFIED_IMAGE="images/tampered_image.jpg"
OUTPUT_DIR="outputs"

# Step 1: Check and Install Dependencies
echo "Checking and installing required Python dependencies..."
if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Please install Python3 to proceed."
    exit 1
fi

if ! command -v pip3 &>/dev/null; then
    echo "pip3 is not installed. Installing pip3..."
    sudo apt-get install -y python3-pip
fi

# Install required Python packages
pip3 install -r requirements.txt --quiet

# Step 2: Create Output Directories (if not already existing)
echo "Ensuring required output directories exist..."
mkdir -p "$OUTPUT_DIR/metadata"
mkdir -p "$OUTPUT_DIR/features"
mkdir -p "$OUTPUT_DIR/ela"
mkdir -p "$OUTPUT_DIR/report"

# Step 3: Run the Python Script
echo "Running the Python script for image analysis..."
if [[ -f "$PYTHON_SCRIPT" ]]; then
    if [[ -f "$ORIGINAL_IMAGE" ]]; then
        if [[ -f "$MODIFIED_IMAGE" ]]; then
            python3 "$PYTHON_SCRIPT" "$ORIGINAL_IMAGE" "$MODIFIED_IMAGE"
        else
            echo "Modified image not found at $MODIFIED_IMAGE. Running for the original image only."
            python3 "$PYTHON_SCRIPT" "$ORIGINAL_IMAGE"
        fi
    else
        echo "Original image not found at $ORIGINAL_IMAGE. Please provide a valid image path."
        exit 1
    fi
else
    echo "Python script $PYTHON_SCRIPT not found. Please ensure it exists in the current directory."
    exit 1
fi

echo "Image analysis completed successfully. Outputs are saved in the '$OUTPUT_DIR' directory."

# Step 4: Preserve Output Directory (Do not delete outputs)
echo "Outputs directory is preserved for future analysis."
