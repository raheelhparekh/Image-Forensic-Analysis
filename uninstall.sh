#!/bin/bash

# Exit on error
set -e

echo "🗑️ Uninstalling dependencies..."

# Uninstall Python dependencies
pip3 uninstall -r requirements.txt -y

echo "✅ Uninstallation complete!"
