import os
import json
import cv2
import numpy as np
from PIL import Image, ImageChops
import exifread
import hashlib
import imagehash

# Create necessary directories
OUTPUT_DIRS = ["outputs/metadata", "outputs/features", "outputs/ela", "outputs/report"]
for directory in OUTPUT_DIRS:
    os.makedirs(directory, exist_ok=True)

def extract_metadata(image_path):
    """
    Extracts EXIF metadata from an image and saves it as a JSON file.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        dict: Extracted metadata.
    """
    try:
        with open(image_path, 'rb') as img_file:
            tags = exifread.process_file(img_file, details=False)

        if not tags:
            print(f"No metadata found in {image_path}")
            return {}

        metadata = {tag: str(tags[tag]) for tag in tags if tag not in ("JPEGThumbnail", "TIFFThumbnail")}

        metadata_filename = os.path.basename(image_path) + "_metadata.json"
        metadata_path = os.path.join("outputs/metadata", metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Metadata extracted and saved to {metadata_path}")
        return metadata

    except Exception as e:
        print(f"Error extracting metadata from {image_path}: {e}")
        return {}

def extract_features(image_path):
    """
    Extracts color histograms, edge maps, and ORB descriptors from an image.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        tuple: (histogram, edge map, ORB descriptors)
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image {image_path}")

        # Color histogram with 16 bins per channel
        hist = cv2.calcHist([image], [0, 1, 2], None, [16, 16, 16], [0, 256] * 3)
        hist = cv2.normalize(hist, hist).flatten()

        # Edge map using Sobel operator
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)

        # ORB Feature extraction
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)

        # Save features
        features_filename = os.path.basename(image_path) + "_features.npz"
        features_path = os.path.join("outputs/features", features_filename)
        np.savez(features_path, hist=hist, edges=sobel_edges, descriptors=descriptors)
        
        print(f"Features extracted and saved to {features_path}")
        return hist, sobel_edges, descriptors

    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None, None, None

def error_level_analysis(image_path, quality=90):
    """
    Performs Error Level Analysis (ELA) on an image and saves the ELA image.

    Parameters:
        image_path (str): Path to the image file.
        quality (int): JPEG quality for compression.

    Returns:
        str: Path to the ELA image.
    """
    try:
        original = Image.open(image_path).convert('RGB')
        compressed_path = os.path.join("outputs/ela", os.path.basename(image_path) + "_compressed.jpg")
        ela_path = os.path.join("outputs/ela", os.path.basename(image_path) + "_ela.jpg")
        
        original.save(compressed_path, 'JPEG', quality=quality)
        compressed = Image.open(compressed_path)

        ela_image = ImageChops.difference(original, compressed)
        ela_image = ela_image.point(lambda p: p * 10)  # Enhance differences
        ela_image.save(ela_path)
        
        print(f"ELA image saved to {ela_path}")
        return ela_path

    except Exception as e:
        print(f"Error performing ELA on {image_path}: {e}")
        return ""

def generate_image_hex(image_path):
    """
    Generates a SHA-256 hex code for the binary content of an image.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        str: Hexadecimal hash of the image.
    """
    try:
        with open(image_path, "rb") as file:
            binary_data = file.read()
        hex_code = hashlib.sha256(binary_data).hexdigest()
        print(f"SHA-256 Hex for {image_path}: {hex_code}")
        return hex_code
    except Exception as e:
        print(f"Error generating SHA-256 hex for {image_path}: {e}")
        return None

def generate_perceptual_hash(image_path, hash_size=16):
    """
    Generates a perceptual hash (phash) for an image.

    Parameters:
        image_path (str): Path to the image file.
        hash_size (int): Size of the hash; higher values increase precision.

    Returns:
        str: Perceptual hash of the image.
    """
    try:
        image = Image.open(image_path)
        phash = imagehash.phash(image, hash_size=hash_size)
        print(f"Perceptual Hash for {image_path}: {phash}")
        return str(phash)
    except Exception as e:
        print(f"Error generating perceptual hash for {image_path}: {e}")
        return None

def compare_hashes(original_hex, modified_hex, original_phash, modified_phash, phash_threshold=5):
    """
    Compares cryptographic and perceptual hashes of two images.

    Parameters:
        original_hex (str): SHA-256 hash of the original image.
        modified_hex (str): SHA-256 hash of the modified image.
        original_phash (str): Perceptual hash of the original image.
        modified_phash (str): Perceptual hash of the modified image.
        phash_threshold (int): Hamming distance threshold for perceptual similarity.

    Returns:
        dict: Comparison results.
    """
    comparison_results = {}

    # Compare SHA-256 hashes
    if original_hex and modified_hex:
        comparison_results['sha256_identical'] = (original_hex == modified_hex)
        if comparison_results['sha256_identical']:
            print("The images are identical at the binary level (SHA-256).")
        else:
            print("The images are different at the binary level (SHA-256).")
            print(f"Original SHA-256: {original_hex}")
            print(f"Modified SHA-256: {modified_hex}")
    else:
        comparison_results['sha256_identical'] = False
        print("SHA-256 comparison failed due to missing hash values.")

    # Compare Perceptual Hashes
    if original_phash and modified_phash:
        original_phash_obj = imagehash.hex_to_hash(original_phash)
        modified_phash_obj = imagehash.hex_to_hash(modified_phash)
        hamming_distance = original_phash_obj - modified_phash_obj
        comparison_results['perceptual_similarity'] = (hamming_distance <= phash_threshold)
        comparison_results['perceptual_hamming_distance'] = hamming_distance

        print(f"Hamming Distance (Perceptual): {hamming_distance}")
        if comparison_results['perceptual_similarity']:
            print("Images are perceptually similar.")
        else:
            print("Images are not perceptually similar.")
    else:
        comparison_results['perceptual_similarity'] = False
        comparison_results['perceptual_hamming_distance'] = None
        print("Perceptual hash comparison failed due to missing hash values.")

    return comparison_results

def generate_report(image_path, metadata, hist, edges, ela_path, sha256_hash, phash, report_suffix="_report.txt", additional_info=None):
    """
    Generates a comprehensive report consolidating analysis results.

    Parameters:
        image_path (str): Path to the image file.
        metadata (dict): Extracted metadata.
        hist (numpy.ndarray): Color histogram.
        edges (numpy.ndarray): Edge map.
        ela_path (str): Path to the ELA image.
        sha256_hash (str): SHA-256 hash of the image.
        phash (str): Perceptual hash of the image.
        report_suffix (str): Suffix for the report filename.
        additional_info (dict): Additional information to include in the report.

    Returns:
        str: Path to the generated report.
    """
    try:
        report_filename = os.path.basename(image_path) + report_suffix
        report_path = os.path.join("outputs/report", report_filename)
        
        with open(report_path, 'w') as f:
            f.write(f"Image Analysis Report for {image_path}\n")
            f.write("="*60 + "\n\n")
            
            f.write("**Metadata:**\n")
            if metadata:
                f.write(json.dumps(metadata, indent=4))
            else:
                f.write("No metadata found.\n")
            f.write("\n\n")
            
            f.write("**Feature Extraction:**\n")
            if hist is not None and edges is not None:
                f.write(f"- Histogram Shape: {hist.shape}\n")
                f.write(f"- Edge Map Shape: {edges.shape}\n")
                f.write(f"- ORB Descriptors: {'Available' if descriptors_available(edges) else 'N/A'}\n")
            else:
                f.write("Feature extraction failed.\n")
            f.write("\n\n")
            
            f.write("**Error Level Analysis (ELA):**\n")
            f.write(f"- ELA Image Path: {ela_path}\n")
            f.write("\n\n")
            
            f.write("**Hash Verification:**\n")
            f.write(f"- SHA-256 Hash: {sha256_hash if sha256_hash else 'N/A'}\n")
            f.write(f"- Perceptual Hash (phash): {phash if phash else 'N/A'}\n")
            f.write("\n\n")
            
            if additional_info:
                f.write("**Additional Information:**\n")
                for key, value in additional_info.items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")
        
        print(f"Report generated at: {report_path}")
        return report_path

    except Exception as e:
        print(f"Error generating report for {image_path}: {e}")
        return ""

def descriptors_available(descriptors):
    """
    Checks if ORB descriptors are available.

    Parameters:
        descriptors (numpy.ndarray): ORB descriptors.

    Returns:
        bool: True if descriptors are available, False otherwise.
    """
    return descriptors is not None

def compare_image_hashes(original_image_path, modified_image_path, phash_threshold=5):
    """
    Compares both cryptographic and perceptual hashes of two images.

    Parameters:
        original_image_path (str): Path to the original image.
        modified_image_path (str): Path to the modified image.
        phash_threshold (int): Threshold for perceptual hash similarity.

    Returns:
        dict: Comparison results.
    """
    original_hex = generate_image_hex(original_image_path)
    modified_hex = generate_image_hex(modified_image_path)

    original_phash = generate_perceptual_hash(original_image_path)
    modified_phash = generate_perceptual_hash(modified_image_path)

    comparison_results = compare_hashes(original_hex, modified_hex, original_phash, modified_phash, phash_threshold)
    return comparison_results

def main(image_path, modified_image_path=None, phash_threshold=5):
    """
    Main function to perform image analysis and hash verification.

    Parameters:
        image_path (str): Path to the original image.
        modified_image_path (str, optional): Path to the modified image. Defaults to None.
        phash_threshold (int, optional): Threshold for perceptual hash similarity. Defaults to 5.
    """
    # Process Original Image
    print(f"\nProcessing Original Image: {image_path}")
    metadata_original = extract_metadata(image_path)
    hist_original, edges_original, descriptors_original = extract_features(image_path)
    ela_path_original = error_level_analysis(image_path)
    sha256_original = generate_image_hex(image_path)
    phash_original = generate_perceptual_hash(image_path)
    report_original = generate_report(
        image_path, metadata_original, hist_original, edges_original,
        ela_path_original, sha256_original, phash_original
    )

    metadata_modified = {}
    report_modified = ""
    comparison_results = {}

    if modified_image_path:
        # Process Modified Image
        print(f"\nProcessing Modified Image: {modified_image_path}")
        metadata_modified = extract_metadata(modified_image_path)
        hist_modified, edges_modified, descriptors_modified = extract_features(modified_image_path)
        ela_path_modified = error_level_analysis(modified_image_path)
        sha256_modified = generate_image_hex(modified_image_path)
        phash_modified = generate_perceptual_hash(modified_image_path)
        report_modified = generate_report(
            modified_image_path, metadata_modified, hist_modified, edges_modified,
            ela_path_modified, sha256_modified, phash_modified
        )

        # Compare Hashes
        print("\nComparing Hashes:")
        comparison_results = compare_image_hashes(
            image_path, modified_image_path, phash_threshold
        )

        # Additional Metadata Comparison
        print("\nComparing Metadata:")
        if metadata_original and not metadata_modified:
            print("Tampering Detected: Original image contains metadata, but modified image does not.")
            # Optionally, you can flag this in the report
            additional_info = {
                "Metadata Comparison": "Original has metadata; Modified lacks metadata."
            }
            # Update the modified report with additional info
            generate_report(
                modified_image_path, metadata_modified, hist_modified, edges_modified,
                ela_path_modified, sha256_modified, phash_modified,
                report_suffix="_report.txt",
                additional_info=additional_info
            )
        elif not metadata_original and metadata_modified:
            print("Note: Original image lacks metadata, but modified image contains metadata.")
        elif not metadata_original and not metadata_modified:
            print("Neither original nor modified images contain metadata.")
        else:
            # Both have metadata; further comparison can be done if needed
            print("Both original and modified images contain metadata.")
            # Optionally, compare specific metadata fields here

        # Summary of Findings
        print("\nSummary of Findings:")
        if metadata_original and not metadata_modified:
            print("• Metadata Discrepancy Detected: Potential Tampering.")
        if not comparison_results.get('sha256_identical', True):
            print("• SHA-256 Hashes Differ: Image has been altered.")
        if not comparison_results.get('perceptual_similarity', True):
            print("• Perceptual Hashes Differ: Visual changes detected.")
        if metadata_original and not metadata_modified:
            print("• Metadata indicates possible tampering due to absence in modified image.")

        # Final Verdict
        if (metadata_original and not metadata_modified) or not comparison_results.get('sha256_identical', True) or not comparison_results.get('perceptual_similarity', True):
            print("\nFinal Verdict: The modified image is likely tampered.")
        else:
            print("\nFinal Verdict: No significant tampering detected.")

    else:
        print(f"Modified image not provided. Only processing the original image.")
        # Optionally, provide insights based solely on the original image

if __name__ == "__main__":
    # Example Usage
    # Replace these paths with your actual image paths
    original_image = "images/cup_image.jpg"         # Path to your original image
    modified_image = "images/tampered_image.jpg"    # Path to your tampered image (optional)

    # Check if original_image exists; if not, prompt the user
    if os.path.exists(original_image):
        if modified_image and os.path.exists(modified_image):
            main(original_image, modified_image, phash_threshold=5)
        else:
            print(f"Modified image not found at {modified_image}. Only processing the original image.")
            main(original_image)
    else:
        print(f"Original image not found at {original_image}. Please check the path.")