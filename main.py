import os
import json
import csv
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
    base_output_dir = "outputs"
    os.makedirs(base_output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(base_output_dir, f"analysis_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    parsed_url = urlparse(website_url)
    domain = parsed_url.netloc or parsed_url.path

    final_filename_json = f"{tool_name}_{user_id}_{timestamp}_{domain}.json"
    final_filename_csv = f"{tool_name}_{user_id}_{timestamp}_{domain}.csv"
    final_output_path_json = os.path.join(run_output_dir, final_filename_json)
    final_output_path_csv = os.path.join(run_output_dir, final_filename_csv)

    return run_output_dir, final_output_path_json, final_output_path_csv

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
        
        # Reduce histogram bins to [8, 8, 8] for fewer values
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hist = cv2.normalize(hist, hist).flatten()
        
        # Compute histogram statistics
        hist_summary = {"mean": float(np.mean(hist)), "std": float(np.std(hist)), "max": float(np.max(hist))}

        # Compute edge maps
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)

        # Compute edge map statistics
        edge_summary = {"mean": float(np.mean(sobel_edges)), "std": float(np.std(sobel_edges)), "max": float(np.max(sobel_edges))}

        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)
        
        # Compute descriptors summary if available
        descriptors_summary = None
        if descriptors is not None:
            descriptors_summary = {"mean": float(np.mean(descriptors)), "std": float(np.std(descriptors)), "max": float(np.max(descriptors))}

        features_filename = os.path.basename(image_path) + "_features.npz"
        features_path = os.path.join(output_dir, features_filename)
        np.savez(features_path, hist=hist, edges=sobel_edges, descriptors=descriptors)
        
        return hist_summary, edge_summary, descriptors_summary
    except Exception as e:
        return None, None, None

def error_level_analysis(image_path, output_dir, quality=90):
    try:
        original = Image.open(image_path).convert('RGB')
        compressed_path = os.path.join(output_dir, os.path.basename(image_path) + "_compressed.jpg")
        ela_path = os.path.join(output_dir, os.path.basename(image_path) + "_ela.jpg")
        
        original.save(compressed_path, 'JPEG', quality=quality)
        compressed = Image.open(compressed_path)

        ela_image = ImageChops.difference(original, compressed)
        ela_image = ela_image.point(lambda p: p * 10)  # Enhance differences
        ela_image.save(ela_path)
        
        return ela_path
    except Exception as e:
        return ""

def save_summary_as_csv(summary, csv_path):
    try:
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Key", "Value"])
            for key, value in summary.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        writer.writerow([f"{key}.{sub_key}", sub_value])
                else:
                    writer.writerow([key, value])
        print(f"✅ Final summary CSV saved to: {csv_path}")
    except Exception as e:
        print(f"❌ Error writing final summary CSV: {e}")

def main(original_image_path, user_id, website_url, tool_name="ImageAnalysisTool"):
    run_output_dir, final_output_path_json, final_output_path_csv = create_output_directories(tool_name, user_id, website_url)

    metadata_original = extract_metadata(original_image_path, run_output_dir)
    hist_summary, edge_summary, descriptors_summary = extract_features(original_image_path, run_output_dir)
    ela_path = error_level_analysis(original_image_path, run_output_dir)

    final_summary = {
        "original_image": {
            "path": original_image_path,
            "metadata": metadata_original,
            "features": {
                "histogram_summary": hist_summary,
                "edge_summary": edge_summary,
                "descriptors_summary": descriptors_summary
            },
            "ela_path": ela_path,
        }
    }

    os.makedirs(os.path.dirname(final_output_path_json), exist_ok=True)
    with open(final_output_path_json, "w") as f:
        json.dump(final_summary, f, indent=4)
    print(f"✅ Final summary JSON saved to: {final_output_path_json}")

    save_summary_as_csv(final_summary, final_output_path_csv)

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
