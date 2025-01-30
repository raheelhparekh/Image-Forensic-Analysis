#!/bin/bash

# Exit on error
set -e

echo "🔄 Installing dependencies for image forensics..."

# Install Python3 if missing
if ! command -v python3 &>/dev/null; then
    echo "⚠️ Python3 is not installed. Installing..."
    sudo apt-get update && sudo apt-get install -y python3 python3-pip
fi

# Ensure pip is up-to-date
pip3 install --upgrade pip setuptools wheel

# Install required Python libraries
pip3 install -r requirements.txt --quiet

# Create output directories if they don't exist
echo "📂 Ensuring output directories exist..."
mkdir -p "outputs/metadata"
mkdir -p "outputs/features"
mkdir -p "outputs/ela"
mkdir -p "outputs/report"

echo "✅ Installation complete!"
