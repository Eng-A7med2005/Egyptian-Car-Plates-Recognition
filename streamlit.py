import streamlit as st
import torch
import numpy as np
import gdown
from PIL import Image
from ultralytics import YOLO

# رابط تحميل النموذج الأول من Google Drive
plate_model_url = 'https://drive.google.com/uc?id=12tRfc_-nOkqMO9bdwpV8P8MFamwgtR2e'

# رابط تحميل النموذج الثاني (يمكنك استبداله برابط محلي إذا كنت تمتلكه)
ocr_model_url = 'path_to_local_ocr_model.pt'  # ضع مسار النموذج الثاني هنا

# مسارات تحميل الملفات
plate_model_path = 'yolo11m_car_plate_trained.pt'
ocr_model_path = 'yolo11m_car_plate_ocr1.pt'

# تنزيل النماذج من Google Drive (النموذج الأول فقط من Google Drive، الثاني إذا كان محليًا)
gdown.download(plate_model_url, plate_model_path, quiet=False)
# إذا كان لديك النموذج الثاني محليًا:
# gdown.download(ocr_model_url, ocr_model_path, quiet=False)

# تحميل النماذج باستخدام YOLO
plate_detector = YOLO(plate_model_path)
ocr_model = YOLO(ocr_model_path)

# قاموس لتحويل الحروف اللاتينية إلى حروف عربية
char_map = {
    'meem': 'م', 'yaa': 'ي', 'alef': 'ا', 'baa': 'ب', 'taa': 'ت', 'thaa': 'ث',
    'jeem': 'ج', 'haa': 'ح', 'khaa': 'خ', 'dal': 'د', 'dhal': 'ذ', 'raa': 'ر',
    'zay': 'ز', 'seen': 'س', 'sheen': 'ش', 'saad': 'ص', 'daad': 'ض', 'ain': 'ع',
    'ghain': 'غ', 'fa': 'ف', 'qaf': 'ق', 'kaf': 'ك', 'lam': 'ل', 'noon': 'ن',
    'ha': 'ه', 'waw': 'و'
}

# قاموس لتحويل الأرقام اللاتينية إلى أرقام عربية
number_map = {
    '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
    '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩',
}

# واجهة Streamlit
st.title("🚘 Car Plate Recognition Using Two YOLOv8 Models")

uploaded_image = st.file_uploader("Upload a Car Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # تحميل الصورة باستخدام PIL بدلاً من OpenCV
    image = Image.open(uploaded_image).convert("RGB")
    rgb_image = np.array(image)

    st.image(rgb_image, caption="Uploaded Image", use_column_width=True)

    # --- اكتشاف اللوحة ---
    plate_results = plate_detector.predict(rgb_image, conf=0.3)

    if len(plate_results[0].boxes) == 0:
        st.warning("No plates were detected.")
    else:
        st.subheader("Detected Plates and Their Content:")
        for i, box in enumerate(plate_results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            plate_crop = rgb_image[y1:y2, x1:x2]

            # --- التعرف على الحروف باستخدام النموذج الثاني ---
            ocr_results = ocr_model.predict(plate_crop, conf=0.25)
            plotted = ocr_results[0].plot()

            # استخراج الحروف والأرقام بترتيبهم
            detected_items = []
            for char_box in ocr_results[0].boxes:
                class_id = int(char_box.cls[0])
                name = ocr_results[0].names[class_id]
                x_center = (char_box.xyxy[0][0] + char_box.xyxy[0][2]) / 2
                detected_items.append((x_center, name))

            detected_items.sort(key=lambda x: x[0])

            # تحويل النص من الحروف اللاتينية إلى العربية
            plate_text = ''.join([char_map.get(item[1], item[1]) for item in detected_items])
            plate_text = ''.join([number_map.get(char, char) for char in plate_text])
            plate_text = ' '.join(reversed(list(plate_text)))

            # عرض النتائج
            st.image(plate_crop, caption=f"Plate Region #{i+1}", use_column_width=True)
            st.image(plotted, caption=f"Characters Detected in Plate #{i+1}", use_column_width=True)
            st.success(f"🔠 Recognized Text: **{plate_text}**")

    # عرض الصورة مع المربعات
    st.image(plate_results[0].plot(), caption="Image with Detected Plates", use_column_width=True)
