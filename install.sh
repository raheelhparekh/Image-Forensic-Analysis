#!/bin/bash

# INSTALL.SH
# Script to set up the environment for the image analysis and tampering detection project

echo "Starting the setup process..."

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Please install Python 3.8 or above."
    exit 1
fi

echo "Python3 is installed. Proceeding with setup..."

# Create and activate a virtual environment
if [ ! -d "venv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists. Skipping creation."
fi

echo "Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate
echo "Virtual environment activated."

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required libraries
echo "Installing required libraries..."
pip install -r requirements.txt

# Create necessary output directories
OUTPUT_DIRS=("outputs/metadata" "outputs/features" "outputs/ela" "outputs/report")
for dir in "${OUTPUT_DIRS[@]}"
do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    else
        echo "Directory $dir already exists. Skipping creation."
    fi
done

echo "Setup completed successfully. You can now run the script using 'python main.py'."