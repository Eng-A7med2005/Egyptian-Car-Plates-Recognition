import os
os.system("apt-get update && apt-get install -y libgl1")

import streamlit as st
import torch
import numpy as np
import gdown
import cv2
from PIL import Image
from ultralytics import YOLO

# روابط Google Drive
plate_model_url = 'https://drive.google.com/uc?id=12tRfc_-nOkqMO9bdwpV8P8MFamwgtR2e'
ocr_model_url = 'https://drive.google.com/uc?id=YOUR_OCR_MODEL_ID'  # ← ضع هنا ID الخاص بالموديل الثاني

# مسارات حفظ الملفات
plate_model_path = 'yolo11m_car_plate_trained.pt'
ocr_model_path = 'yolo11m_car_plate_ocr1.pt'

# تحميل النماذج
gdown.download(plate_model_url, plate_model_path, quiet=False)
gdown.download(ocr_model_url, ocr_model_path, quiet=False)

# تحميل الموديلات
plate_detector = YOLO(plate_model_path)
ocr_model = YOLO(ocr_model_path)

# قاموس تحويل الحروف
char_map = {
    'alef': 'ا', 'baa': 'ب', 'taa': 'ت', 'thaa': 'ث', 'jeem': 'ج', 'haa': 'ح', 'khaa': 'خ',
    'dal': 'د', 'dhal': 'ذ', 'raa': 'ر', 'zay': 'ز', 'seen': 'س', 'sheen': 'ش', 'saad': 'ص',
    'daad': 'ض', 'taa2': 'ط', 'thaa2': 'ظ', 'ain': 'ع', 'ghain': 'غ', 'fa': 'ف', 'qaf': 'ق',
    'kaf': 'ك', 'lam': 'ل', 'meem': 'م', 'noon': 'ن', 'ha': 'ه', 'waw': 'و', 'yaa': 'ي'
}

number_map = {
    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
    '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'
}

# واجهة Streamlit
st.title("🚘 Egyptian Car Plate Recognition")

uploaded_image = st.file_uploader("Upload a Car Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(rgb_image, caption="Uploaded Image", use_column_width=True)

    # كشف اللوحات
    plate_results = plate_detector.predict(rgb_image, conf=0.3)

    if len(plate_results[0].boxes) == 0:
        st.warning("لم يتم اكتشاف أي لوحة.")
    else:
        st.subheader("اللوحات التي تم اكتشافها:")
        for i, box in enumerate(plate_results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            plate_crop = rgb_image[y1:y2, x1:x2]

            # التعرف على الأحرف
            ocr_results = ocr_model.predict(plate_crop, conf=0.25)
            plotted = ocr_results[0].plot()

            detected_items = []
            for char_box in ocr_results[0].boxes:
                class_id = int(char_box.cls[0])
                name = ocr_results[0].names[class_id]
                x_center = (char_box.xyxy[0][0] + char_box.xyxy[0][2]) / 2
                detected_items.append((x_center, name))

            detected_items.sort(key=lambda x: x[0])

            plate_text = ''.join([char_map.get(item[1], item[1]) for item in detected_items])
            plate_text = ''.join([number_map.get(char, char) for char in plate_text])
            plate_text = ' '.join(reversed(plate_text))

            st.image(plate_crop, caption=f"Plate #{i+1}", use_column_width=True)
            st.image(plotted, caption=f"Characters Detected", use_column_width=True)
            st.success(f"🔠 Recognized Text: **{plate_text}**")

    st.image(plate_results[0].plot(), caption="Image with Detected Plates", use_column_width=True)
