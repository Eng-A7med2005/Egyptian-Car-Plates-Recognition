# 🚗 Egyptian Car Plate Recognition System | Machathon 3.0

## 🔍 Problem Statement

- Collect and/or generate sufficient data  
- Detect and extract license plates  
- Recognize characters (Arabic and Latin letters, Eastern Arabic numerals)
- Deliver a fast, accurate, and easy-to-deploy solution

## 🧠 Solution Overview

Our system is built in two main stages:

### 1. License Plate Detection (Object Detection)
- **Model**: YOLOv8
- **Trained on**: 6000 images (1000 real + 5000 generated)
- **Task**: Detect the license plate region in the car image

### 2. Character Recognition (OCR)
- **Model**: EasyOCR + custom post-processing
- **Languages Supported**: Arabic + English
- **Post-processing**: Clean up output and map Eastern Arabic numerals to Western

### 🔁 Pipeline Flow
