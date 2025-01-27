#!/bin/bash

# Automation Script for Image Tampering Detection
# Author: Your Name

# Directory and file paths
PROJECT_DIR=$(pwd)
PYTHON_SCRIPT="$PROJECT_DIR/main.py"
IMAGE1="images/image.jpeg"       # First image path
IMAGE2="images/image2.jpeg"      # Second image path
OUTPUT_DIR="$PROJECT_DIR/outputs"

# Step 1: Install Dependencies
install_dependencies() {
    echo "Checking and installing dependencies..."
    
    # Ensure Python and pip are installed
    if ! command -v python3 &> /dev/null; then
        echo "Python3 not found. Installing Python3..."
        sudo apt update
        sudo apt install python3 -y
    fi

    if ! command -v pip3 &> /dev/null; then
        echo "pip3 not found. Installing pip3..."
        sudo apt install python3-pip -y
    fi

    # Install required Python libraries
    echo "Installing required Python packages..."
    pip3 install -r requirements.txt
    echo "Dependencies installed successfully."
}

# Step 2: Run Python Script
run_script() {
    echo "Running the Python script..."
    python3 "$PYTHON_SCRIPT" "$IMAGE1" "$IMAGE2"
    echo "Script execution completed."
}

# Step 3: Display Results
display_results() {
    echo "Displaying outputs..."
    
    # Display metadata files
    echo "Metadata extracted:"
    ls "$OUTPUT_DIR/metadata"

    # Display reports
    echo "Generated reports:"
    ls "$OUTPUT_DIR/report"

    echo "Outputs saved in: $OUTPUT_DIR"
}

# Step 4: Cleanup Outputs (Optional)
cleanup_outputs() {
    echo "Cleaning up outputs..."
    rm -rf "$OUTPUT_DIR"
    echo "Outputs removed."
}

# Menu to guide user
main_menu() {
    echo "======================================"
    echo " Image Tampering Detection Automation "
    echo "======================================"
    echo "1. Install Dependencies"
    echo "2. Run Analysis Script"
    echo "3. View Results"
    echo "4. Cleanup Outputs"
    echo "5. Exit"
    echo "======================================"
    read -p "Choose an option [1-5]: " choice

    case $choice in
        1)
            install_dependencies
            ;;
        2)
            run_script
            ;;
        3)
            display_results
            ;;
        4)
            cleanup_outputs
            ;;
        5)
            echo "Exiting script. Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid option. Please try again."
            main_menu
            ;;
    esac
}

# Run the menu
main_menu
