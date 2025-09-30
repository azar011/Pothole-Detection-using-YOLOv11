
# Pothole Detection Using YOLOv11 🚧

An AI-powered real-time pothole detection system using **YOLOv11**, integrated with **Streamlit**, and enhanced with **image, video, and smartphone camera detection**. The system also performs **repair material estimation** to assist smart infrastructure maintenance.

---

## 🧠 Project Overview

This project aims to improve road safety and optimize maintenance workflows by identifying potholes in real-time from various sources including static images, uploaded videos, and live smartphone feeds. The YOLOv11 model ensures high accuracy and speed for object detection, while the user-friendly web interface supports practical deployment and public interaction.

---

## 📂 Dataset

- **Source**: [Roboflow Dataset – Pothole Detection](https://universe.roboflow.com/project-ssayl/potholes-detection-d4rma/dataset/1)
- **Format**: YOLOv11-compatible with train, validation, and test splits
- **Annotations**: Bounding boxes for pothole instances
- **License**: CC BY 4.0

---

## 💡 Features

- 🔍 **YOLOv11-based Pothole Detection** (Enhanced speed & accuracy)
- 🖼️ **Image Upload Detection**
- 📹 **Real-Time Video Processing**
- 📱 **Smartphone Camera Integration**
- 📏 **Material & Cost Estimation**
- 📊 **Detection Summary with Downloadable Report**
- 🌐 **Streamlit Web UI for Visualization & Interaction**

---

## 🏗️ Project Structure

```
├── app.py                        # Streamlit frontend
├── detect.py                     # YOLOv11 detection wrapper
├── model metrics.ipynb           # For model evalution
├── Weights/
│   └── best.pt                   # Trained YOLOv11 model weights
├── requirements.txt              # Python dependencies
└── README.md                     # Project overview
```

---

## 🖥️ Tech Stack

- 🔍 YOLOv11 (Ultralytics)
- 🐍 Python 3.12
- 💻 OpenCV
- 📈 NumPy & Pandas
- 🎨 Streamlit
- 📦 Roboflow for Dataset Management

---

## 🛠️ Installation

```bash
git clone https://github.com/arul0076/pothole-detection-yolov11-streamlit.git
cd pothole-detection-yolov11-streamlit
pip install -r requirements.txt
```

---

## 🚀 Run the App

```bash
streamlit run app.py
```

Upload an image or video, or use your phone camera to start detecting potholes!

---
![Screenshot 2025-03-27 155546](https://github.com/user-attachments/assets/451612bb-5bbe-433d-9fc5-6b5f5cf4642d)

## 📐 Material Estimation Logic

- Materials calculated: **Tar**, **Sand**, **Small Stones**
- Based on total damaged area in square meters
- Costs are estimated using predefined price per sqm
- Report includes quantity & total cost for repair

---

## 📸 Sample Outputs

- 📷 Image Detection
- 🎞️ Video Detection
- 📱 Live Camera Detection
- 📊 Auto-generated Detection Report (CSV)

---

## 📈 Results

- **mAP@50**: 87.3%
- **Precision**: 91.7%
- **Recall**: 77.6%
- Tested on 1.8k+ images (Roboflow validation set)

---

## 📚 References

1. YOLOv11 official implementation by Ultralytics  
2. Roboflow dataset and annotation tools  
3. Real-time computer vision techniques (OpenCV)  
4. Streamlit app deployment  

---

## 🤝 Contributors

- **Arul Palaniappa S**  
- **Azarudeen B**  
- **Balaji C**

---

## 📜 License

This project is licensed under the MIT License.
