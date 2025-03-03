# Evidence Tampering Detection

This project is designed to analyze images for evidence tampering and generate detailed reports on the findings. It uses various image processing techniques to extract metadata, compute hashes, and perform error level analysis (ELA).

## Features

- Extract EXIF metadata from images
- Compute SHA-256 and perceptual hashes (pHash) for image integrity verification
- Generate histograms, edge maps, and ORB descriptors for image features
- Perform Error Level Analysis (ELA) to detect tampering
- Save analysis results in JSON and CSV formats

## Requirements

- Python 3.8 or above
- Required Python libraries (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Make the `install.sh` script executable and run it to set up the environment:
    ```sh
    chmod +x install.sh
    ./install.sh
    ```

## Usage

### Running the Analysis

1. Make the `run_analysis.sh` script executable:
    ```sh
    chmod +x run_analysis.sh
    ```

2. Run the script with the required parameters:
    ```sh
    ./run_analysis.sh <path-to-original-image> <path-to-tampered-image> <user-id> <website-url>
    ```

    Example:
    ```sh
    ./run_analysis.sh images/sample_original.jpg images/sample_tampered.jpg test_user example.com
    ```

### Running the Automation Script

1. Make the `automation.sh` script executable:
    ```sh
    chmod +x automation.sh
    ```

2. Run the script and follow the prompts:
    ```sh
    ./automation.sh
    ```

### Uninstallation

To clean up the environment, run the `uninstall.sh` script:
```sh
chmod +x uninstall.sh
./uninstall.sh
```