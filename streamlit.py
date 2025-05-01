import gdown
import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# تحميل النموذج الأول: لاكتشاف اللوحات
plate_model_url = 'https://drive.google.com/uc?id=12tRfc_-nOkqMO9bdwpV8P8MFamwgtR2e'
plate_model_file = 'yolo11m_car_plate_ocr.pt'
gdown.download(plate_model_url, plate_model_file, quiet=False)
plate_detector = YOLO(plate_model_file)

# تحميل النموذج الثاني: لقراءة محتوى اللوحة
ocr_model_file = 'yolo11m_car_plate_ocr1.pt'
ocr_model = YOLO(ocr_model_file)

# واجهة Streamlit
st.title("🚘 Car Plate Recognition Using Dual YOLO Models")

uploaded_image = st.file_uploader("Upload a Car Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(rgb_image, caption="Uploaded Image", use_column_width=True)

    # ---- الخطوة 1: اكتشاف اللوحات في الصورة ----
    plate_results = plate_detector.predict(rgb_image, conf=0.3)

    if len(plate_results[0].boxes) == 0:
        st.warning("No plates were detected.")
    else:
        st.subheader("Detected Plates and Their Content:")
        for i, box in enumerate(plate_results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            plate_crop = rgb_image[y1:y2, x1:x2]

            # ---- الخطوة 2: قراءة محتوى اللوحة باستخدام النموذج الثاني ----
            ocr_results = ocr_model.predict(plate_crop, conf=0.25)

            # رسم الصورة مع المربعات
            plotted = ocr_results[0].plot()

            # استخراج الحروف والأرقام حسب ترتيبهم من اليسار لليمين
            detected_items = []
            for char_box in ocr_results[0].boxes:
                class_id = int(char_box.cls[0])
                name = ocr_results[0].names[class_id]
                x_center = (char_box.xyxy[0][0] + char_box.xyxy[0][2]) / 2
                detected_items.append((x_center, name))

            # ترتيب العناصر حسب X
            detected_items.sort(key=lambda x: x[0])
            plate_text = ''.join([item[1] for item in detected_items])

            # ---- العرض ----
            st.image(plate_crop, caption=f"Plate Region #{i+1}", use_column_width=True)
            st.image(plotted, caption=f"Characters Detected in Plate #{i+1}", use_column_width=True)
            st.success(f"🔠 Recognized Text: **{plate_text}**")

    # عرض الصورة العامة مع اللوحات
    img_with_boxes = plate_results[0].plot()
    st.image(img_with_boxes, caption="Image with Detected Plates", use_column_width=True)
