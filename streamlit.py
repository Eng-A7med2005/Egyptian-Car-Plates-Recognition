import gdown
import os
import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO

# =======================
# تحميل النموذج من Google Drive لو مش موجود
# =======================
model_path = 'yolo11m_car_plate_ocr.pt'

if not os.path.exists(model_path):
    url = 'https://drive.google.com/file/d/12tRfc_-nOkqMO9bdwpV8P8MFamwgtR2e'  # <-- غيّر الـ ID هنا
    st.write("⏳ Downloading model...")
    gdown.download(url, model_path, quiet=False)
    st.success("✅ Model downloaded successfully!")

# =======================
# تحميل النموذج
# =======================
model = YOLO(model_path)

# =======================
# Streamlit UI
# =======================
st.title("🚘 Arabic Car Plate Recognition")
uploaded_file = st.file_uploader("Upload a car image", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # حفظ مؤقت للصورة
    temp_path = 'temp_image.jpg'
    image.save(temp_path)

    # التنبؤ
    results = model.predict(source=temp_path, conf=0.25)
    result_image = results[0].plot()

    # عرض النتائج
    st.image(result_image, caption="Detected Car Plate", use_column_width=True)
