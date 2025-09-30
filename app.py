# app.py
import streamlit as st
import cv2
import tempfile
import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pothole_detection import detect_potholes, process_video, process_realtime_video, calculate_materials, MATERIAL_COSTS, COST_PER_SQM


# Set page config
st.set_page_config(
    page_title="AI Pothole Detection System",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        background: linear-gradient(to bottom right, #0e1117, #1e2130);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .status-card {
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .metric-container {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        background-color: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        padding: 15px;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header {
        text-align: center;
        margin-bottom: 30px;
    }
    .header h1 {
        color: #4CAF50;
    }
    .header p {
        color: #cccccc;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: #cccccc;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="header">
    <h1>🚧 AI-Powered Pothole Detection System</h1>
    <p>YOLOv11 advanced computer vision for road maintenance</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processed_file_path' not in st.session_state:
    st.session_state.processed_file_path = None

# Sidebar
with st.sidebar:
    st.title("Detection Settings")
    
    # Input selection - limited to the three requested options
    option = st.radio("Select Input Type:", ("Image", "Real-Time Video", "Phone Camera"))
    
    # Detection settings
    st.subheader("Detection Parameters")
    confidence = st.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.05)
    
    # Cost settings
    st.subheader("Cost Parameters")
    cost_per_sqm = st.number_input("Cost per sqm (₹)", min_value=100, max_value=1000, value=300, step=50)
    
    # Custom footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 AI Pothole Detection System</p>
        <p>Built with YOLOv11 and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# Main content based on selection
if option == "Image":
    st.subheader("Upload an Image for Pothole Detection")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Save uploaded file
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_img.write(uploaded_file.read())
        temp_img_path = temp_img.name
        temp_img.close()
        
        # Read image
        img = cv2.imread(temp_img_path)
        
        # Create columns for before/after
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="status-card">', unsafe_allow_html=True)
            st.image(temp_img_path, caption="Uploaded Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Process image
        with st.spinner("Detecting potholes..."):
            processed_img, pothole_count, sqm_area, repair_cost, potholes = detect_potholes(img)
            
            # Convert BGR to RGB for display
            processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown('<div class="status-card">', unsafe_allow_html=True)
                st.image(processed_img_rgb, caption="Processed Image with Detections", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Results and metrics
        st.markdown("## 📊 Detection Results")
        
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Potholes Detected", pothole_count, delta=f"{pothole_count} issues")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_cols[1]:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Damage Area", f"{sqm_area:.2f} sqm", delta=f"{sqm_area:.1f} sqm")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_cols[2]:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Repair Cost", f"₹{repair_cost:.2f}", delta=f"₹{repair_cost:.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with metric_cols[3]:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Road Quality", f"{max(0, 100 - pothole_count*5):.0f}%", delta=f"-{pothole_count*5}%", delta_color="inverse")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Material calculation
        st.markdown("## 🧱 Required Materials")
        
        materials = calculate_materials(sqm_area)
        total_material_cost = sum(qty * MATERIAL_COSTS.get(mat, 0) for mat, qty in materials.items())
        
        material_cols = st.columns(len(materials) + 1)
        
        for i, (material, qty) in enumerate(materials.items()):
            with material_cols[i]:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                cost = qty * MATERIAL_COSTS.get(material, 0)
                st.metric(material, f"{qty:.1f}", delta=f"₹{cost:.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with material_cols[-1]:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Total Material Cost", f"₹{total_material_cost:.2f}", delta=f"₹{total_material_cost:.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Clean up
        os.unlink(temp_img_path)
        
elif option == "Real-Time Video":
    st.subheader("📹 Real-Time Video Processing")

    uploaded_file = st.file_uploader("📂 Choose a video for real-time processing...", type=["mp4", "avi", "mov"])

    if uploaded_file:
        # Save uploaded file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name
        temp_video.close()

        # Display original video
        st.video(temp_video_path)

        # ✅ Initialize session state
        if "stop_video" not in st.session_state:
            st.session_state.stop_video = False
        if "video_running" not in st.session_state:
            st.session_state.video_running = False
        if "final_pothole_count" not in st.session_state:
            st.session_state.final_pothole_count = 0

        # ✅ Create placeholders
        stframe = st.empty()
        stop_button_container = st.empty()

        # ✅ Start Processing
        if st.button("▶️ Start Real-Time Analysis"):
            st.session_state.video_running = True
            st.session_state.stop_video = False  
            cap = cv2.VideoCapture(temp_video_path)

            if not cap.isOpened():
                st.error("⚠️ Error: Unable to open video file.")
            else:
                detected_potholes = set()  # ✅ Track unique potholes
                frame_count = 0
                fps = int(cap.get(cv2.CAP_PROP_FPS))  # ✅ Get video FPS
                last_detections = []  # ✅ Store last detected boxes for smooth rendering

                # ✅ Show Stop button only when video starts
                with stop_button_container.container():
                    if st.button("🛑 Stop Video Processing"):
                        st.session_state.stop_video = True

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or st.session_state.stop_video:
                        break

                    frame_count += 1

                    # ✅ Process every 5th frame (Increases speed)
                    if frame_count % 5 == 0:
                        frame, pothole_count, _, _, pothole_data = detect_potholes(frame)

                        # ✅ Store latest bounding box detections
                        last_detections = [(pothole["coords"], pothole["confidence"]) for pothole in pothole_data]

                        # ✅ Track unique potholes without duplicate counting
                        for pothole in pothole_data:
                            detected_potholes.add(tuple(pothole["coords"]))

                    # ✅ Draw last detected bounding boxes on video (Stable boxes)
                    for (x1, y1, x2, y2), conf in last_detections:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # ✅ Show processed video frame inside webpage (Smooth playback)
                    stframe.image(frame, channels="BGR", use_container_width=True)

                    # ✅ Maintain normal playback speed
                    time.sleep(1 / fps)  

                cap.release()

                # ✅ Store final pothole count
                st.session_state.final_pothole_count = len(detected_potholes)
                st.session_state.video_running = False  # Ensure results persist

        # ✅ Show final summary **after clicking Stop**
        if not st.session_state.video_running and st.session_state.final_pothole_count > 0:
            st.markdown("## 📊 Final Detection Summary")
            st.metric("Total Potholes Detected", st.session_state.final_pothole_count)


elif option == "Phone Camera":
    st.subheader("📱 Real-Time Phone Camera Detection")
    st.info("Connect your phone camera to this computer using an IP Webcam app.")

    # Camera settings
    cam_url = st.text_input("Enter Phone Camera URL (e.g., http://192.168.X.X:4747/video)", "http://192.168.1.10:4747/video")

    # Connection instructions
    with st.expander("📱 How to connect your phone camera"):
        st.markdown("""
        1. Install 'IP Webcam' app on your Android phone (available on Google Play Store)
        2. Connect your phone to the same WiFi network as this computer
        3. Open the app and scroll down to 'Start Server'
        4. Look for the IP address displayed at the bottom of your phone screen (e.g., http://192.168.1.10:4747)
        5. Enter this address in the text field above
        6. Click 'Start Live Detection' below
        """)

    # ✅ Initialize stop flag in session state
    if "stop_realtime" not in st.session_state:
        st.session_state.stop_realtime = False

    # Start camera feed
    if st.button("🎥 Start Live Detection"):
        st.session_state.stop_realtime = False  # Reset stop flag
        cap = cv2.VideoCapture(cam_url)

        if not cap.isOpened():
            st.error("⚠️ Error: Unable to access phone camera. Check the IP.")
        else:
            frame_placeholder = st.empty()  # Placeholder for displaying frames
            stop_button = st.button("🛑 Stop Detection")  # Stop button

            while cap.isOpened() and not st.session_state.stop_realtime:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process real-time frames
                frame, detected_potholes, sqm_area, repair_cost, _ = detect_potholes(frame)

                # Display processed frames in Streamlit
                frame_placeholder.image(frame, channels="BGR")

            cap.release()

    # ✅ STOP button functionality
    if st.button("🛑 Stop Detection"):
        st.session_state.stop_realtime = True
        st.success("✅ Live Detection Stopped Successfully!")

# Add short documentation section
st.markdown("---")
with st.expander("ℹ️ About this System"):
    st.markdown("""
    ### AI Pothole Detection System
    
    This system uses YOLOv11, a state-of-the-art computer vision model, to detect and analyze potholes in roads.
    
    **Features:**
    - Detect potholes in images with high accuracy
    - Real-time analysis of video feeds
    - Phone camera integration for field inspections
    - Automatic area measurement and repair cost estimation
    - Material requirement calculation for road maintenance
    
    **How it works:**
    1. The system uses a YOLOv11 model trained on a dataset of road images
    2. Potholes are detected and their dimensions are calculated
    3. Repair costs and material requirements are estimated based on the area
    4. Results are visualized in an interactive dashboard
    """)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 AI Pothole Detection System</p>
</div>
""", unsafe_allow_html=True)