import gdown
import streamlit as st
import cv2
import torch
import numpy as np  # إضافة numpy هنا
from ultralytics import YOLO

# تحميل النموذج من Google Drive باستخدام gdown
url = 'https://drive.google.com/uc?id=12tRfc_-nOkqMO9bdwpV8P8MFamwgtR2e'
output = 'yolo11m_car_plate_ocr.pt'
gdown.download(url, output, quiet=False)

# تحميل النموذج المدرب من الملف
model = YOLO('yolo11m_car_plate_ocr.pt')

# دالة لمعالجة الصورة
def process_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # تحويل الصورة من BGR إلى RGB
    return img

# واجهة رفع الصورة في Streamlit
st.title("Car Plate Number Recognition")
uploaded_image = st.file_uploader("Upload a Car Image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # قراءة الصورة
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
    processed_image = process_image(image)

    # عرض الصورة الأصلية
    st.image(processed_image, caption="Uploaded Image", use_column_width=True)

    # تنفيذ التنبؤ باستخدام YOLO على الصورة المرفوعة
    results = model.predict(processed_image, conf=0.25)

    # عرض النتائج (رقم اللوحة)
    st.subheader("Detected Plate Number:")
    for result in results[0].names:  # عرض أسماء الكائنات المكتشفة
        st.write(f"Plate: {result}")

    # رسم المربعات حول اللوحات على الصورة
    img_with_boxes = results[0].plot()

    # عرض الصورة مع المربعات المرسومة
    st.image(img_with_boxes, caption="Image with Detected Plate", use_column_width=True)
