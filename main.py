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
from skimage.metrics import structural_similarity as ssim


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
        with open(image_path, "rb") as img_file:
            tags = exifread.process_file(img_file, details=False)
        metadata = {
            tag: str(tags[tag])
            for tag in tags
            if tag not in ("JPEGThumbnail", "TIFFThumbnail")
        }

        metadata_filename = os.path.basename(image_path) + "_metadata.json"
        metadata_path = os.path.join(output_dir, metadata_filename)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return metadata
    except Exception as e:
        return {}


def compute_hashes(image_path):
    try:
        with open(image_path, "rb") as file:
            sha256_hash = hashlib.sha256(file.read()).hexdigest()

        image = Image.open(image_path)
        phash = str(imagehash.phash(image))

        return {"sha256": sha256_hash, "phash": phash}
    except Exception as e:
        return {"sha256": None, "phash": None}


def extract_features(image_path, output_dir):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Reduce histogram bins to [8, 8, 8] for fewer values
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hist = cv2.normalize(hist, hist).flatten()

        # Compute histogram statistics
        hist_summary = {
            "mean": float(np.mean(hist)),
            "std": float(np.std(hist)),
            "max": float(np.max(hist)),
        }

        # Compute edge maps
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges = cv2.magnitude(sobel_x, sobel_y)

        # Compute edge map statistics
        edge_summary = {
            "mean": float(np.mean(sobel_edges)),
            "std": float(np.std(sobel_edges)),
            "max": float(np.max(sobel_edges)),
        }

        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)

        # Compute descriptors summary if available
        descriptors_summary = None
        if descriptors is not None:
            descriptors_summary = {
                "mean": float(np.mean(descriptors)),
                "std": float(np.std(descriptors)),
                "max": float(np.max(descriptors)),
            }

        features_filename = os.path.basename(image_path) + "_features.npz"
        features_path = os.path.join(output_dir, features_filename)
        np.savez(features_path, hist=hist, edges=sobel_edges, descriptors=descriptors)

        return hist_summary, edge_summary, descriptors_summary
    except Exception as e:
        return None, None, None


def error_level_analysis(image_path, output_dir, quality=90):
    try:
        original = Image.open(image_path).convert("RGB")
        compressed_path = os.path.join(
            output_dir, os.path.basename(image_path) + "_compressed.jpg"
        )
        ela_path = os.path.join(output_dir, os.path.basename(image_path) + "_ela.jpg")

        original.save(compressed_path, "JPEG", quality=quality)
        compressed = Image.open(compressed_path)

        ela_image = ImageChops.difference(original, compressed)
        ela_image = ela_image.point(lambda p: p * 10)  # Enhance differences
        ela_image.save(ela_path)

        return ela_path
    except Exception as e:
        return ""


def compare_images(original_image_path, tampered_image_path, output_dir):
    comparison_results = {}

    # Load images
    original_image = cv2.imread(original_image_path)
    tampered_image = cv2.imread(tampered_image_path)

    # Resize images to the same dimensions
    if original_image.shape != tampered_image.shape:
        print(
            "⚠️ Images have different dimensions. Resizing tampered image to match original image dimensions."
        )
        tampered_image = cv2.resize(
            tampered_image, (original_image.shape[1], original_image.shape[0])
        )

    # Compare metadata
    metadata_original = extract_metadata(original_image_path, output_dir)
    metadata_tampered = extract_metadata(tampered_image_path, output_dir)
    comparison_results["metadata_differences"] = {
        k: (metadata_original.get(k), metadata_tampered.get(k))
        for k in set(metadata_original) | set(metadata_tampered)
        if metadata_original.get(k) != metadata_tampered.get(k)
    }

    # Compare hashes
    hashes_original = compute_hashes(original_image_path)
    hashes_tampered = compute_hashes(tampered_image_path)
    comparison_results["hash_differences"] = {
        k: (hashes_original.get(k), hashes_tampered.get(k))
        for k in set(hashes_original) | set(hashes_tampered)
        if hashes_original.get(k) != hashes_tampered.get(k)
    }

    # Compare histograms, edges, and descriptors
    hist_summary_original, edge_summary_original, descriptors_summary_original = (
        extract_features(original_image_path, output_dir)
    )
    hist_summary_tampered, edge_summary_tampered, descriptors_summary_tampered = (
        extract_features(tampered_image_path, output_dir)
    )
    comparison_results["feature_differences"] = {
        "histogram": {
            k: (hist_summary_original.get(k), hist_summary_tampered.get(k))
            for k in set(hist_summary_original) | set(hist_summary_tampered)
            if hist_summary_original.get(k) != hist_summary_tampered.get(k)
        },
        "edges": {
            k: (edge_summary_original.get(k), edge_summary_tampered.get(k))
            for k in set(edge_summary_original) | set(edge_summary_tampered)
            if edge_summary_original.get(k) != edge_summary_tampered.get(k)
        },
        "descriptors": {
            k: (
                descriptors_summary_original.get(k),
                descriptors_summary_tampered.get(k),
            )
            for k in set(descriptors_summary_original)
            | set(descriptors_summary_tampered)
            if descriptors_summary_original.get(k)
            != descriptors_summary_tampered.get(k)
        },
    }

    # Compare ELA
    ela_original = error_level_analysis(original_image_path, output_dir)
    ela_tampered = error_level_analysis(tampered_image_path, output_dir)

    if ela_original and ela_tampered:
        with open(ela_original, "rb") as f1, open(ela_tampered, "rb") as f2:
            if f1.read() == f2.read():
                comparison_results["ela_differences"] = None
            else:
                comparison_results["ela_differences"] = {
                    "original_ela_path": ela_original,
                    "tampered_ela_path": ela_tampered,
                }
    else:
        comparison_results["ela_differences"] = None

    # Compare SSIM
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    tampered_gray = cv2.cvtColor(tampered_image, cv2.COLOR_BGR2GRAY)
    ssim_index, diff = ssim(original_gray, tampered_gray, full=True)
    comparison_results["ssim"] = {
        "index": ssim_index,
        "diff_path": os.path.join(output_dir, "ssim_diff.jpg"),
    }
    cv2.imwrite(comparison_results["ssim"]["diff_path"], (diff * 255).astype(np.uint8))

    # Pixel-wise difference
    diff_image = cv2.absdiff(original_image, tampered_image)
    diff_image_path = os.path.join(output_dir, "pixel_diff.jpg")
    cv2.imwrite(diff_image_path, diff_image)
    comparison_results["pixel_differences"] = {"diff_path": diff_image_path}

    # Add a human-readable summary
    comparison_results["summary"] = {
        "ssim_interpretation": {
            "description": "Structural Similarity Index (SSIM) measures how similar two images are. A value of 1 means the images are identical, while a value closer to 0 means they are very different.",
            "value": ssim_index,
            "interpretation": f"The SSIM value of {ssim_index:.2f} indicates {'high similarity' if ssim_index > 0.9 else 'moderate similarity' if ssim_index > 0.7 else 'low similarity'} between the original and tampered images.",
        },
        "pixel_differences_interpretation": {
            "description": "Pixel-wise difference shows the absolute difference between the two images. Bright areas in the difference image indicate regions where the images differ significantly.",
            "interpretation": "Check the 'pixel_diff.jpg' image to see where the original and tampered images differ. Bright areas indicate tampering or changes.",
        },
        "ela_interpretation": {
            "description": "Error Level Analysis (ELA) highlights areas of an image that have been compressed differently. These areas may indicate tampering.",
            "interpretation": "Check the 'original_ela.jpg' and 'tampered_ela.jpg' images. Bright areas in the ELA images suggest regions that may have been altered.",
        },
        "conclusion": {
            "description": "Based on the analysis, the tampered image shows differences in the following areas:",
            "differences": [
                diff
                for diff in [
                    (
                        "Metadata differences"
                        if comparison_results.get("metadata_differences")
                        else None
                    ),
                    (
                        "Hash differences"
                        if comparison_results.get("hash_differences")
                        else None
                    ),
                    (
                        "Feature differences"
                        if any(
                            comparison_results.get("feature_differences", {}).values()
                        )
                        else None
                    ),
                    (
                        "ELA differences"
                        if comparison_results.get("ela_differences")
                        else None
                    ),
                    "SSIM differences" if ssim_index < 1.0 else None,
                    "Pixel differences" if np.any(diff_image > 0) else None,
                ]
                if diff is not None
            ],
            "final_verdict": (
                "The image appears to be altered."
                if any(
                    [
                        comparison_results.get("metadata_differences"),
                        comparison_results.get("hash_differences"),
                        any(comparison_results.get("feature_differences", {}).values()),
                        comparison_results.get("ela_differences"),
                        ssim_index < 1.0,
                        np.any(diff_image > 0),
                    ]
                )
                else "The image appears to be identical to the original."
            ),
        },
    }

    return comparison_results


def save_summary_as_csv(summary, csv_path):
    try:
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Key", "Value"])
            for key, value in summary.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        writer.writerow([f"{key}.{sub_key}", sub_value])
                else:
                    writer.writerow([key, value])
        print(f" Final summary CSV saved to: {csv_path}")
    except Exception as e:
        print(f" Error writing final summary CSV: {e}")


def main(
    original_image_path,
    tampered_image_path,
    user_id,
    website_url,
    tool_name="ImageAnalysisTool",
):
    run_output_dir, final_output_path_json, final_output_path_csv = (
        create_output_directories(tool_name, user_id, website_url)
    )

    metadata_original = extract_metadata(original_image_path, run_output_dir)
    metadata_tampered = extract_metadata(tampered_image_path, run_output_dir)
    hashes_original = compute_hashes(original_image_path)
    hashes_tampered = compute_hashes(tampered_image_path)
    hist_summary_original, edge_summary_original, descriptors_summary_original = (
        extract_features(original_image_path, run_output_dir)
    )
    hist_summary_tampered, edge_summary_tampered, descriptors_summary_tampered = (
        extract_features(tampered_image_path, run_output_dir)
    )
    ela_original = error_level_analysis(original_image_path, run_output_dir)
    ela_tampered = error_level_analysis(tampered_image_path, run_output_dir)
    comparison_results = compare_images(
        original_image_path, tampered_image_path, run_output_dir
    )

    final_summary = {
        "original_image": {
            "path": original_image_path,
            "metadata": metadata_original,
            "hashes": hashes_original,
            "features": {
                "histogram_summary": hist_summary_original,
                "edge_summary": edge_summary_original,
                "descriptors_summary": descriptors_summary_original,
            },
            "ela_path": ela_original,
        },
        "tampered_image": {
            "path": tampered_image_path,
            "metadata": metadata_tampered,
            "hashes": hashes_tampered,
            "features": {
                "histogram_summary": hist_summary_tampered,
                "edge_summary": edge_summary_tampered,
                "descriptors_summary": descriptors_summary_tampered,
            },
            "ela_path": ela_tampered,
        },
        "comparison_results": comparison_results,
    }

    os.makedirs(os.path.dirname(final_output_path_json), exist_ok=True)
    with open(final_output_path_json, "w") as f:
        json.dump(final_summary, f, indent=4)
    print(f"✅ Final summary JSON saved to: {final_output_path_json}")

    save_summary_as_csv(final_summary, final_output_path_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Image Forensics")
    parser.add_argument("--original", required=True, help="Path to original image")
    parser.add_argument("--tampered", required=True, help="Path to tampered image")
    parser.add_argument("--user", required=True, help="User ID for output naming")
    parser.add_argument("--website", required=True, help="Website URL for naming")
    parser.add_argument("--tool", default="ImageAnalysisTool", help="Tool name")

    args = parser.parse_args()

    main(
        original_image_path=args.original,
        tampered_image_path=args.tampered,
        user_id=args.user,
        website_url=args.website,
        tool_name=args.tool,
    )
