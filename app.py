import streamlit as st
import pandas as pd
from io import BytesIO
from main import extract_metadata, extract_features, generate_image_hex, generate_perceptual_hash

# Streamlit setup
st.title("Image Analysis Tool")
st.markdown("""
### Analyze your images for metadata, features, and hash generation.
Upload an image to start the analysis.
""")

# File upload
uploaded_file = st.file_uploader("Upload an image (JPG or JPEG only)", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = f"images/{uploaded_file.name}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(temp_file_path, caption="Uploaded Image", use_container_width=True)


    # Results dictionary to store all extracted data
    results = {}

    # Metadata Extraction
    st.write("### Metadata Extraction:")
    metadata = extract_metadata(temp_file_path)
    if metadata:
        st.json(metadata)
        results["Metadata"] = metadata
    else:
        st.write("No metadata found.")

    # Feature Extraction
    st.write("### Feature Extraction:")
    hist, edges, descriptors = extract_features(temp_file_path)
    if hist is not None and edges is not None:
        st.write(f"Histogram Shape: {hist.shape}")
        st.write(f"Edge Map Shape: {edges.shape}")
        st.write(f"ORB Descriptors: {'Available' if descriptors is not None else 'Not Available'}")
        results["Histogram Shape"] = hist.shape
        results["Edge Map Shape"] = edges.shape
        results["ORB Descriptors"] = "Available" if descriptors is not None else "Not Available"
    else:
        st.write("Failed to extract features.")

    # Hash Generation
    st.write("### Hash Generation:")
    sha256_hash = generate_image_hex(temp_file_path)
    perceptual_hash = generate_perceptual_hash(temp_file_path)
    st.write(f"SHA-256 Hash: {sha256_hash}")
    st.write(f"Perceptual Hash: {perceptual_hash}")
    results["SHA-256 Hash"] = sha256_hash
    results["Perceptual Hash"] = perceptual_hash

    # Download Results as Excel
    st.write("### Download Results")
    if st.button("Download Results as Excel"):
        # Convert results dictionary into a pandas DataFrame
        metadata_df = pd.DataFrame(metadata.items(), columns=["Metadata Key", "Metadata Value"]) if metadata else pd.DataFrame()
        features_df = pd.DataFrame({
            "Feature": ["Histogram Shape", "Edge Map Shape", "ORB Descriptors"],
            "Value": [hist.shape if hist is not None else "N/A", edges.shape if edges is not None else "N/A", "Available" if descriptors is not None else "Not Available"]
        })
        hash_df = pd.DataFrame({
            "Hash Type": ["SHA-256 Hash", "Perceptual Hash"],
            "Hash Value": [sha256_hash, perceptual_hash]
        })

        # Write to Excel using BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            metadata_df.to_excel(writer, index=False, sheet_name="Metadata")
            features_df.to_excel(writer, index=False, sheet_name="Features")
            hash_df.to_excel(writer, index=False, sheet_name="Hashes")
        output.seek(0)

        # Download button
        st.download_button(
            label="Download Excel",
            data=output,
            file_name="image_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
