#!/bin/bash

# UNINSTALL.SH
# Script to clean up the environment for the image analysis and tampering detection project

echo "Starting the uninstallation process..."

# Deactivate virtual environment if active
if [[ "$VIRTUAL_ENV" != "" ]]
then
    echo "Deactivating virtual environment..."
    deactivate
fi

# Remove virtual environment
if [ -d "venv" ]; then
    echo "Removing virtual environment..."
    rm -rf venv
    echo "Virtual environment removed."
else
    echo "No virtual environment found. Skipping removal."
fi

# Optionally, remove Python caches or temporary files
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +

# Keep output directories but notify the user
echo "Output directories are preserved. If you wish to delete them, remove the 'outputs' directory manually."

echo "Uninstallation completed successfully."