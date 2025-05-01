import gdown
import streamlit as st
import cv2
import torch
import numpy as np
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
    st.image(processed_image, caption="Uploaded Image", use_container_width=True)

    # تنفيذ التنبؤ باستخدام YOLO على الصورة المرفوعة
    results = model.predict(processed_image, conf=0.25)

    # استخراج الأرقام من اللوحة (OCR) إذا كانت هناك لوحات مكتشفة
    st.subheader("Detected Plate Numbers:")
    for box in results[0].boxes:
        # استخراج إحداثيات كل مربع تم اكتشافه
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cropped_plate = processed_image[y1:y2, x1:x2]  # قص الجزء الذي يحتوي على اللوحة

        # يمكن هنا استخدام نموذج OCR لاستخراج النص من الصورة المقصوصة (إذا كنت تستخدم Tesseract أو غيره)
        # مثال لاستخدام Tesseract (تحتاج إلى تثبيت Tesseract)
        # import pytesseract
        # plate_text = pytesseract.image_to_string(cropped_plate, config='--psm 8')

        # هنا يمكن عرض الأرقام المكتشفة
        st.image(cropped_plate, caption="Detected Plate Region", use_container_width=True)
        st.write("Plate Number: (Here you would display the OCR result)")

    # رسم المربعات حول اللوحات على الصورة
    img_with_boxes = results[0].plot()

    # عرض الصورة مع المربعات المرسومة
    st.image(img_with_boxes, caption="Image with Detected Plate", use_container_width=True)
