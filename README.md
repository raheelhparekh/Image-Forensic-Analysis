# Evidence Tampering Detection

This project is designed to analyze images for evidence tampering and generate detailed reports on the findings. It uses various image processing techniques to extract metadata, compute hashes, and perform error level analysis (ELA).

## Features

- Extract EXIF metadata from images
- Compute SHA-256 and perceptual hashes (pHash) for image integrity verification
- Generate histograms, edge maps, and ORB descriptors for image features
- Perform Error Level Analysis (ELA) to detect tampering
- Calculate Structural Similarity Index (SSIM) between images
- Compute pixel-wise differences between images
- Highlight tampered areas using grid-based analysis
- Generate heatmaps to visualize tampered areas
- Create side-by-side comparison images
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

## Project Flow

1. **Setup Environment**: The `install.sh` script sets up the environment by creating a virtual environment, installing required libraries, and creating necessary output directories.

2. **Run Analysis**: The `run_analysis.sh` script runs the image analysis by executing the `main.py` script with the provided image paths, user ID, and website URL.

3. **Automation**: The `automation.sh` script automates the process by prompting the user for inputs and running the `main.py` script with those inputs.

4. **Uninstallation**: The `uninstall.sh` script cleans up the environment by deactivating and removing the virtual environment and optionally removing Python cache files.

## Inputs

- **Original Image Path**: Path to the original image.
- **Tampered Image Path**: Path to the tampered image.
- **User ID**: User ID for output naming.
- **Website URL**: Website URL for naming.

## Outputs

- **Metadata**: Extracted EXIF metadata from images.
- **Hashes**: SHA-256 and perceptual hashes (pHash) for image integrity verification.
- **Features**: Histograms, edge maps, and ORB descriptors for image features.
- **ELA**: Error Level Analysis (ELA) images to detect tampering.
- **SSIM**: Structural Similarity Index (SSIM) value and difference image.
- **Pixel-wise Differences**: Image showing pixel-wise differences between the original and tampered images.
- **Highlighted Tampered Areas**: Image highlighting areas where tampering is detected.
- **Heatmap**: Heatmap visualizing tampered areas.
- **Side-by-Side Comparison**: Image showing the original, tampered, and highlighted images side by side.
- **Comparison Results**: Detailed comparison results between the original and tampered images.
- **Summary**: Human-readable summary of the analysis results.
- **JSON and CSV Files**: Analysis results saved in JSON and CSV formats.

## How It Works

1. **Extract Metadata**: The `extract_metadata` function extracts EXIF metadata from the images and saves it in JSON format.

2. **Compute Hashes**: The `compute_hashes` function computes SHA-256 and perceptual hashes (pHash) for image integrity verification.

3. **Extract Features**: The `extract_features` function generates histograms, edge maps, and ORB descriptors for image features.

4. **Error Level Analysis (ELA)**: The `error_level_analysis` function performs ELA to detect tampering by highlighting areas of an image that have been compressed differently.

5. **Compare Images**: The `compare_images` function compares the original and tampered images by analyzing metadata, hashes, features, ELA, SSIM, and pixel-wise differences.

6. **Highlight Tampered Areas**: The `highlight_tampered_areas` function highlights areas where tampering is detected based on grid division and generates a heatmap.

7. **Generate Summary**: The `main` function generates a human-readable summary of the analysis results and saves it in JSON and CSV formats.

## Example

To run the analysis, use the following command:
```sh
./run_analysis.sh images/sample_original.jpg images/sample_tampered.jpg test_user example.com
```

This will generate the analysis results and save them in the `outputs` directory.

## Conclusion

The Evidence Tampering Detection project provides a comprehensive solution for analyzing images for evidence tampering. It uses various image processing techniques to extract metadata, compute hashes, and perform error level analysis (ELA), and generates detailed reports on the findings.