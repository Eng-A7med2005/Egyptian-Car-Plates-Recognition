import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# تحميل النموذجين من الملفات المحلية
plate_model_path = 'yolo11m_car_plate_trained.pt'
ocr_model_path = 'yolo11m_car_plate_ocr1.pt'

plate_detector = YOLO(plate_model_path)
ocr_model = YOLO(ocr_model_path)

# قاموس لتحويل الحروف اللاتينية إلى حروف عربية
char_map = {
    'meem': 'م',  # meem -> م
    'yaa': 'ي',   # yaa -> ي
    'alef': 'ا',  # alef -> ا
    'baa': 'ب',   # baa -> ب
    'taa': 'ت',   # taa -> ت
    'thaa': 'ث',  # thaa -> ث
    'jeem': 'ج',  # jeem -> ج
    'haa': 'ح',   # haa -> ح
    'khaa': 'خ',  # khaa -> خ
    'daal': 'د',   # dal -> د
    'dhal': 'ذ',  # dhal -> ذ
    'raa': 'ر',   # raa -> ر
    'zay': 'ز',   # zay -> ز
    'seen': 'س',  # seen -> س
    'sheen': 'ش', # sheen -> ش
    'saad': 'ص',  # saad -> ص
    'daad': 'ض',  # daad -> ض
    'taa': 'ط',   # taa -> ط
    'thaa': 'ظ',  # thaa -> ظ
    'ain': 'ع',   # ain -> ع
    'ghain': 'غ', # ghain -> غ
    'faa': 'ف',    # fa -> ف
    'qaaf': 'ق',   # qaf -> ق
    'kaf': 'ك',   # kaf -> ك
    'laam': 'ل',   # lam -> ل
    'meem': 'م',  # meem -> م
    'noon': 'ن',  # noon -> ن
    'haa': 'ه',    # ha -> ه
    'waaw': 'و',   # waw -> و
    'yaa': 'ي'    # yaa -> ي
}

# قاموس لتحويل الأرقام اللاتينية إلى أرقام عربية
number_map = {
    '0': '٠',
    '1': '١',
    '2': '٢',
    '3': '٣',
    '4': '٤',
    '5': '٥',
    '6': '٦',
    '7': '٧',
    '8': '٨',
    '9': '٩',
}

# واجهة Streamlit
st.markdown(
    """
    <h1 style='text-align: center;'>🚘 Car Plate Recognition Using Two YOLOv8 Models</h1>
    <h3 style='text-align: center;'>By AIvolution Team</h3>
    """,
    unsafe_allow_html=True
)

uploaded_image = st.file_uploader("Upload a Car Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

            # استخدام القاموس لتحويل النص من الحروف اللاتينية إلى الحروف العربية
            plate_text = ''.join([char_map.get(item[1], item[1]) for item in detected_items])

            # تحويل الأرقام إلى أرقام عربية
            plate_text = ''.join([number_map.get(char, char) for char in plate_text])

            # إضافة مسافة بين كل حرف ورقم
            plate_text = ' '.join([char for char in plate_text])

            # عكس النص بحيث يظهر بشكل صحيح
            plate_text = ' '.join(reversed(plate_text.split()))

            # عرض النتائج
            st.image(plate_crop, caption=f"Plate Region #{i+1}", use_column_width=True)
            st.image(plotted, caption=f"Characters Detected in Plate #{i+1}", use_column_width=True)
            st.success(f"🔠 Recognized Text: **{plate_text}**") 

    # عرض الصورة مع المربعات
    st.image(plate_results[0].plot(), caption="Image with Detected Plates", use_column_width=True)
