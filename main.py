import os
import json
import cv2
import numpy as np
from PIL import Image, ImageChops
import exifread
import hashlib
import imagehash
from datetime import datetime
from urllib.parse import urlparse
import argparse

def create_output_directories(tool_name, user_id, website_url):
    """
    Creates a main outputs folder and a timestamped subfolder inside it.
    Also generates the final output filename based on the given requirements.
    Returns:
        tuple: (run_output_dir, final_output_path)
    """
    base_output_dir = "outputs"
    os.makedirs(base_output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(base_output_dir, f"analysis_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    parsed_url = urlparse(website_url)
    domain = parsed_url.netloc or parsed_url.path

    final_filename = f"{tool_name}_{user_id}_{timestamp}_{domain}.json"
    final_output_path = os.path.join(run_output_dir, final_filename)

    return run_output_dir, final_output_path

def extract_metadata(image_path, output_dir):
    try:
        with open(image_path, 'rb') as img_file:
            tags = exifread.process_file(img_file, details=False)
        metadata = {tag: str(tags[tag]) for tag in tags if tag not in ("JPEGThumbnail", "TIFFThumbnail")}

        metadata_filename = os.path.basename(image_path) + "_metadata.json"
        metadata_path = os.path.join(output_dir, metadata_filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        return metadata
    except Exception as e:
        return {}

def extract_features(image_path, output_dir):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([image], [0, 1, 2], None, [16, 16, 16], [0, 256] * 3)
        hist = cv2.normalize(hist, hist).flatten()

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)

        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)

        features_filename = os.path.basename(image_path) + "_features.npz"
        features_path = os.path.join(output_dir, features_filename)
        np.savez(features_path, hist=hist, edges=sobel_edges, descriptors=descriptors)
        
        return hist, sobel_edges, descriptors
    except Exception as e:
        return None, None, None

def generate_image_hash(image_path):
    try:
        with open(image_path, "rb") as file:
            return hashlib.sha256(file.read()).hexdigest()
    except Exception as e:
        return None

def generate_perceptual_hash(image_path, hash_size=16):
    try:
        image = Image.open(image_path)
        return str(imagehash.phash(image, hash_size=hash_size))
    except Exception as e:
        return None

def main(original_image_path, user_id, website_url, tool_name="ImageAnalysisTool"):
    run_output_dir, final_output_path = create_output_directories(tool_name, user_id, website_url)

    metadata_original = extract_metadata(original_image_path, run_output_dir)
    hist_original, edges_original, descriptors_original = extract_features(original_image_path, run_output_dir)
    sha256_original = generate_image_hash(original_image_path)
    phash_original = generate_perceptual_hash(original_image_path)

    final_summary = {
        "original_image": {
            "path": original_image_path,
            "metadata": metadata_original,
            "sha256": sha256_original,
            "phash": phash_original,
        }
    }

    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    with open(final_output_path, "w") as f:
        json.dump(final_summary, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Image Forensics")
    parser.add_argument('--original', required=True, help="Path to original image")
    parser.add_argument('--user', required=True, help="User ID for output naming")
    parser.add_argument('--website', required=True, help="Website URL for naming")
    parser.add_argument('--tool', default="ImageAnalysisTool", help="Tool name")
    
    args = parser.parse_args()
    
    main(
        original_image_path=args.original,
        user_id=args.user,
        website_url=args.website,
        tool_name=args.tool
    )