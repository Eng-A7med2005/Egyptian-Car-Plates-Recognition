# **Streamlit WEB**
LINK : https://egyptian-car-plates-recognition-aivolution-team.streamlit.app/

Egyptian Car Plate Recognition System

# 🔍 Problem Statement  
- Collect and/or generate sufficient data  
- Detect and extract license plates  
- Recognize characters (Arabic and Latin letters, Eastern Arabic numerals)  
- Deliver a fast, accurate, and easy-to-deploy solution  

# 🧠 Solution Overview  
Our system is built in two main stages:

1. **License Plate Detection (Object Detection)**  
   - **Model**: YOLOv8  
   - **Trained on**: 6000 images (1000 real + 5000 generated)  
   - **Task**: Detect the license plate region in the car image  

2. **Character Recognition (OCR)**  
   - **Model**: EasyOCR + custom post-processing  
   - **Languages Supported**: Arabic + English  
   - **Post-processing**: Clean up output and map Eastern Arabic numerals to Western  

# 🔁 Pipeline Flow  
`Input Image → YOLOv8 Detection → Plate Cropping → EasyOCR → Final Text Output`

# 📦 Libraries Used  
- `numpy` – Numerical operations and array handling  
- `pandas` – Data manipulation and analysis  
- `matplotlib`, `seaborn`, `plotly.express` – Visualization of data and results  
- `os`, `shutil`, `yaml`, `warnings`, `random` – File system operations, config loading, and general utilities  
- `cv2 (OpenCV)` – Image processing and license plate extraction  
- `PIL (Pillow)` – Image file handling and manipulation  
- `ultralytics` – YOLOv8 implementation for object detection  
- `easyocr` – OCR tool that supports Arabic and English character recognition  
- `wandb` – Tracking experiments, visualizing model metrics, and collaboration
