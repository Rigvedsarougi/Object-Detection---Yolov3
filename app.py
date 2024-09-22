import streamlit as st
import numpy as np
import cv2
from utils import load_model, detect_objects

# Title and description
st.title("Object Detection with YOLOv3")
st.write("Upload an image, and the YOLOv3 model will detect objects in it.")

# Sidebar options
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
nms_threshold = st.sidebar.slider("Non-Max Suppression Threshold", 0.0, 1.0, 0.3)

# Load the YOLOv3 model
model, output_layers, labels = load_model()

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Perform object detection
    detections = detect_objects(img, model, output_layers, confidence_threshold, nms_threshold)

    # Draw bounding boxes and labels on the image
    for (class_id, confidence, box) in detections:
        x, y, w, h = box
        label = str(labels[class_id])
        color = (0, 255, 0)  # Green box
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the result
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
