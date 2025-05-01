import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr

# تحميل نموذج YOLO المدرب لاكتشاف لوحات السيارات
model = YOLO('yolo11m_car_plate_trained.pt')

# تحميل قارئ EasyOCR مع اللغة العربية والإنجليزية
reader = easyocr.Reader(['ar', 'en'])

st.set_page_config(page_title="قراءة لوحة السيارة", layout="centered")

st.title("📸 التعرف على لوحة أرقام السيارة")
st.markdown("**قم برفع صورة لسيارة وسنقوم بقراءة اللوحة آليًا باستخدام الذكاء الاصطناعي.**")

uploaded_file = st.file_uploader("📤 ارفع صورة السيارة", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="📷 الصورة الأصلية", use_column_width=True)

    # تشغيل نموذج YOLO لاكتشاف لوحة السيارة
    results = model.predict(source=image_np, conf=0.3)
    boxes = results[0].boxes

    if boxes:
        st.markdown("### ✅ تم اكتشاف لوحة!")
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = image_np[y1:y2, x1:x2]

            # عرض اللوحة المقطوعة
            st.image(plate_crop, caption="🔍 لوحة السيارة", use_column_width=False)

            # قراءة النص من اللوحة باستخدام EasyOCR
            text_result = reader.readtext(plate_crop)
            if text_result:
                detected_text = text_result[0][-2]
                st.markdown(f"### 🔤 الرقم المقروء: `{detected_text}`")
            else:
                st.warning("لم يتمكن النموذج من قراءة النص بوضوح. جرب صورة أوضح.")
    else:
        st.warning("❌ لم يتم العثور على لوحة في الصورة.")
