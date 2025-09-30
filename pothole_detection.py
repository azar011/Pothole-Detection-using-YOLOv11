# pothole_detection.py
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime

# Load YOLO model
model = YOLO("Weights/best.pt")

# Cost & Material Estimation
COST_PER_SQM = 300  # Repair cost per square meter
MATERIALS_PER_SQM = {
    "Tar (liters)": 10,
    "Sand (kg)": 50,
    "Small Stones (kg)": 40
}
MATERIAL_COSTS = {
    "Tar (liters)": 10,  # ₹ per liter
    "Sand (kg)": 1.5,    # ₹ per kg
    "Small Stones (kg)": 2  # ₹ per kg
}

# Detection Thresholds
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.4

def process_detections(img, results):
    """Processes detections and calculates pothole data."""
    pothole_count = 0
    total_area = 0
    potholes = []
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = round(box.conf[0].item(), 2)
            if conf > CONFIDENCE_THRESHOLD:
                pothole_count += 1
                area = (x2 - x1) * (y2 - y1)
                total_area += area
                potholes.append({
                    "coords": (x1, y1, x2, y2),
                    "area": area,
                    "confidence": conf
                })
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with confidence and area
                label = f"Pothole: {conf:.2f}, Area: {area/100:.1f}cm²"
                cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img, pothole_count, total_area, potholes

def calculate_materials(sqm_area):
    """Estimate materials needed for repair based on pothole area."""
    return {mat: sqm_area * qty for mat, qty in MATERIALS_PER_SQM.items()}

def detect_potholes(image):
    """Detect potholes in an image and calculate repair cost."""
    img = cv2.resize(image, (640, 360))
    with torch.no_grad():
        results = model(img, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)
    
    processed_img, pothole_count, total_area, potholes = process_detections(img, results)
    
    sqm_area = total_area / 10000 if pothole_count > 0 else 0
    repair_cost = sqm_area * COST_PER_SQM
    
    # Add overlay with summary stats
    overlay = processed_img.copy()
    cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, processed_img, 0.4, 0, processed_img)
    cv2.putText(processed_img, f"Potholes: {pothole_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(processed_img, f"Area: {sqm_area:.2f} sqm", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(processed_img, f"Cost: ₹{repair_cost:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return processed_img, pothole_count, sqm_area, repair_cost, potholes

def process_video(video_path):
    """Process video file, detect potholes, and return processed video path."""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"processed_video_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 360))
    
    # Analysis variables
    pothole_count = 0
    total_area = 0
    frame_count = 0
    unique_potholes = []
    frame_potholes = []
    start_time = time.time()
    
    # Process each frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        
        # Process every 3rd frame to improve speed
        if frame_count % 3 == 0:  
            processed_frame, detected_potholes, sqm_area, _, frame_pothole_data = detect_potholes(frame)
            pothole_count += detected_potholes
            total_area += sqm_area
            frame_potholes.append({"frame": frame_count, "potholes": frame_pothole_data})
            
            # Add progress bar
            cv2.rectangle(processed_frame, (10, height-30), (width-10, height-20), (0, 0, 0), -1)
            progress_width = int((width-20) * (frame_count / total_frames))
            cv2.rectangle(processed_frame, (10, height-30), (10 + progress_width, height-20), (0, 255, 0), -1)
            cv2.putText(processed_frame, f"Processing: {progress:.1f}%", (10, height-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            out.write(processed_frame)
        else:
            # Skip frames for faster processing but still write to output
            resized_frame = cv2.resize(frame, (640, 360))
            out.write(resized_frame)
    
    # Calculate processing metrics
    elapsed_time = time.time() - start_time
    fps_processing = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    # Cleanup
    cap.release()
    out.release()
    
    # Calculate repair cost
    repair_cost = total_area * COST_PER_SQM
    materials = calculate_materials(total_area)
    
    # Create analysis report
    report = {
        "pothole_count": pothole_count,
        "total_area": total_area,
        "repair_cost": repair_cost,
        "materials": materials,
        "processing_time": elapsed_time,
        "fps": fps_processing,
        "frame_potholes": frame_potholes
    }
    
    return output_path, report

def process_realtime_video(video_path=None, cam_url=None, processing_mode="file"):
    """Process video in real-time with continuous feedback."""
    if processing_mode == "file" and video_path:
        cap = cv2.VideoCapture(video_path)
    elif processing_mode == "camera" and cam_url:
        cap = cv2.VideoCapture(cam_url)
    else:
        return "Error: Invalid processing mode or missing path"
    
    if not cap.isOpened():
        return "Error: Unable to open video source"
    
    # Output setup if processing a file
    output_path = None
    out = None
    if processing_mode == "file":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"processed_video_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 360))
    
    # Analysis variables
    pothole_count = 0
    total_area = 0
    frame_count = 0
    start_time = time.time()
    last_update_time = start_time
    update_interval = 1.0  # Update stats every second
    running_stats = []
    
    # Create window with resizable feature
    window_name = "Pothole Detection - Real-time Analysis"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        current_time = time.time()
        
        # Process every 2nd frame to maintain real-time performance
        if frame_count % 2 == 0:
            processed_frame, detected_potholes, sqm_area, cost, _ = detect_potholes(frame)
            
            # Update running statistics
            if detected_potholes > 0:
                pothole_count += detected_potholes
                total_area += sqm_area
                running_stats.append({"time": current_time - start_time, 
                                     "potholes": detected_potholes, 
                                     "area": sqm_area})
            
            # Add elapsed time
            elapsed = current_time - start_time
            cv2.putText(processed_frame, f"Time: {elapsed:.1f}s", 
                       (processed_frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add real-time FPS counter
            fps = frame_count / (current_time - start_time)
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", 
                       (processed_frame.shape[1] - 150, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow(window_name, processed_frame)
            
            # Write to output if processing a file
            if processing_mode == "file" and out is not None:
                out.write(processed_frame)
                
        # Update cumulative stats display every second
        if current_time - last_update_time >= update_interval:
            last_update_time = current_time
            print(f"\rElapsed: {current_time-start_time:.1f}s | Potholes: {pothole_count} | Area: {total_area:.2f} sqm | Cost: ₹{total_area * COST_PER_SQM:.2f}", end="")
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Final stats
    total_time = time.time() - start_time
    final_report = {
        "pothole_count": pothole_count,
        "total_area": total_area,
        "repair_cost": total_area * COST_PER_SQM,
        "materials": calculate_materials(total_area),
        "processing_time": total_time,
        "fps": frame_count / total_time if total_time > 0 else 0,
        "running_stats": running_stats
    }
    
    # Cleanup
    cap.release()
    if out is not None:
        out.write(processed_frame)
        out.release()
    cv2.destroyAllWindows()
    
    return output_path, final_report