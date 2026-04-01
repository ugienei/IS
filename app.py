import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import Counter

st.set_page_config(page_title="Traffic Signs Detection System", layout="wide")

# ===== CONFIG =====
CONF_THRESHOLD = 0.5
IMG_SIZE = 64
DATASET_PATH = "dataset"

# ===== LABEL MAP =====
label_map = {
    "speed_limit": "🚗 Speed Limit",
    "stop": "🛑 Stop",
    "left": "⬅️ Left",
    "right": "➡️ Right"
}

# ===== SAMPLE IMAGES =====
sample_images = {
    "Stop": "ex_pic/stop1.jpg",
    "Left": "ex_pic/left1.jpg",
    "Right": "ex_pic/right1.jpg",
    "Speed Limit 1": "ex_pic/speed_limit1.jpg",
    "Speed Limit 2": "ex_pic/speed_limit2.jpg",
    "9": "ex_pic/9.jpg"
}

# ===== LOAD YOLO =====
try:
    model = YOLO("models/best_yolo.pt")
except:
    model = YOLO("yolov8n.pt")

# ===== AUTO CONF =====
def auto_confidence(results):
    confs = []
    for r in results:
        for box in r.boxes:
            confs.append(float(box.conf[0]))

    if len(confs) == 0:
        return 0.3

    avg_conf = sum(confs) / len(confs)
    return max(0.2, avg_conf - 0.25)

# ===== LOAD ML MODEL =====
@st.cache_resource
def train_ml():
    X, y = [], []

    for label in os.listdir(DATASET_PATH):
        folder = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img.flatten())
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf = RandomForestClassifier()
    svm = SVC(probability=True)
    knn = KNeighborsClassifier(n_neighbors=min(3, len(X_train)))

    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    return rf, svm, knn

rf, svm, knn = train_ml()

# ===== ML PREDICT =====
def ml_predict(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.flatten().reshape(1, -1)

    p1 = rf.predict(img)[0]
    p2 = svm.predict(img)[0]
    p3 = knn.predict(img)[0]

    votes = [p1, p2, p3]
    vote_count = Counter(votes)

    final = vote_count.most_common(1)[0][0]
    return final, vote_count

# ===== UI =====
st.title("🚦 Traffic Signs Detection System")
st.caption("YOLO Detection + ML Ensemble Classification")

# ===== PAGE SELECT =====
page = st.radio(
    "📌 Select Page",
    ["Detection", "About", "Machine Learning", "Neural Network"],
    horizontal=True
)

# ================= DETECTION =================
if page == "Detection":

    mode = st.radio("Select Mode", ["Auto", "Manual"], horizontal=True)

    if mode == "Manual":
        confidence_level = st.selectbox("🎚️ Confidence Level", ["Low", "Medium", "High"])

        if confidence_level == "Low":
            CONF_THRESHOLD = 0.3
        elif confidence_level == "Medium":
            CONF_THRESHOLD = 0.6
        else:
            CONF_THRESHOLD = 0.8

    show_box = st.toggle("Show Bounding Box", value=True)

    uploaded = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

    sample_choice = st.selectbox(
        "📁 Or choose sample image",
        ["None"] + list(sample_images.keys())
    )

    img = None

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
    elif sample_choice != "None":
        img = cv2.imread(sample_images[sample_choice])

    if img is not None:
        original_img = img.copy()
        result_img = img.copy()

        temp_results = model(result_img, conf=0.5)

        if mode == "Auto":
            CONF_THRESHOLD = auto_confidence(temp_results)

        results = model(result_img, conf=CONF_THRESHOLD, imgsz=640)

        # fallback
        if len(results[0].boxes) == 0:
            results = model(result_img, conf=0.2, imgsz=640)

        st.info(f"Confidence = {CONF_THRESHOLD:.2f}")

        if show_box:
            for r in results:
                for box in r.boxes:
                    conf_val = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(result_img, (x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(result_img, f"{label} {conf_val:.2f}",
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,(0,255,0),2)

        ml_result, vote_count = ml_predict(original_img)
        pretty_label = label_map.get(ml_result, ml_result)

        col1, col2 = st.columns(2)

        with col1:
            st.image(original_img, channels="BGR", caption="Original")

        with col2:
            st.image(result_img, channels="BGR", caption="Result")

        # st.success(f"Prediction: {pretty_label}")

# ================= ABOUT PAGE =================
elif page == "About":

    st.header("📘 System Development & Dataset")

    # ===== DATASET =====
    st.subheader("📊 Dataset Source")

    st.markdown("""
**Dataset:** GTSRB (German Traffic Sign Recognition Benchmark)  
🔗 https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign  

Dataset นี้ถูกใช้ในงาน Computer Vision สำหรับจำแนกป้ายจราจร  
เป็นภาพจริงจากถนนในประเทศเยอรมนี
    """)

    st.subheader("📦 Dataset Overview")
    st.markdown("""
- มากกว่า **40 Classes**
- เป็นภาพสีแบบ **RGB**
- ขนาดประมาณ **32x32 pixels**
- เป็นปัญหาแบบ **Multi-class Classification**
    """)

    st.subheader("🧩 Dataset Features")
    st.markdown("""
- 🖼️ ภาพ RGB (3 channels)  
- 🏷️ มี label เป็น class ของป้ายจราจร  
- 🌍 ความหลากหลายสูง (แสง มุม กล้อง สภาพอากาศ)  
- 🎯 1 ภาพ = 1 ป้าย  
    """)

    st.divider()

    # ===== DEVELOPMENT =====
    st.subheader("⚙️ System Development Process")

    st.markdown("""
**1. Data Preparation**
- Resize ภาพเป็น 64x64  
- แปลงภาพเป็น vector สำหรับ ML  

**2. Model Development**
- 🧠 Neural Network: YOLO สำหรับ Object Detection  
- 🤖 Machine Learning: Random Forest, SVM, KNN  

**3. Ensemble Technique**
- ใช้ Majority Voting รวมผลจาก ML  

    """)

    st.divider()

    # ===== ALGORITHMS =====
    st.subheader("🧠 Algorithms Used")

    st.markdown("""
**YOLO (You Only Look Once)**
- ตรวจจับวัตถุแบบ real-time  
- ให้ทั้งตำแหน่ง (bounding box) และ class  

**Random Forest**
- ใช้ decision tree หลายต้น  
- ลด overfitting  

**SVM**
- แยกข้อมูลด้วย hyperplane  
- เหมาะกับ classification  

**KNN**
- ใช้ข้อมูลใกล้เคียงในการตัดสินใจ  
    """)

    st.divider()

    # ===== FEATURES =====
    st.subheader("✨ System Features")

    st.markdown("""
- 🤖 Auto Confidence  
- 🎚️ Manual Confidence  
- 📦 Toggle Bounding Box  
- 🖼️ Sample Image Testing  
    """)

    st.divider()

    st.success("""
ระบบนี้ใช้ทั้ง Neural Network และ Machine Learning  
เพื่อเปรียบเทียบและเพิ่มประสิทธิภาพในการจำแนกป้ายจราจร
    """)

# ================= ML PAGE =================
elif page == "Machine Learning":

    st.header("🤖 โมเดล Machine Learning")

    st.write("""
ระบบนี้ใช้ Machine Learning สำหรับจำแนกประเภทป้ายจราจร
โดยใช้ภาพที่ผ่านการแปลงเป็นข้อมูลตัวเลข
    """)

    st.subheader("โมเดลที่ใช้")
    st.write("""
- Random Forest
- SVM
- KNN
    """)

    st.subheader("หลักการทำงาน")
    st.write("""
- Resize ภาพเป็น 64x64
- แปลงเป็น vector
- ใช้ Majority Voting รวมผล
    """)

    st.subheader("ข้อดี")
    st.write("""
- เร็ว
- ใช้ข้อมูลน้อย
    """)

    st.subheader("ข้อจำกัด")
    st.write("""
- ไม่สามารถหาตำแหน่งวัตถุได้
    """)

# ================= YOLO PAGE =================
elif page == "Neural Network":

    st.header("🧠 Neural Network (YOLOv8)")

    st.write("""
YOLO เป็น Deep Learning ที่ใช้ตรวจจับวัตถุในภาพ
สามารถระบุทั้งตำแหน่งและประเภทของป้ายจราจร
    """)

    st.subheader("หลักการทำงาน")
    st.write("""
- แบ่งภาพเป็น grid
- หา bounding box
- ทำนาย class
    """)

    st.subheader("ข้อดี")
    st.write("""
- ตรวจจับได้แม่น
- Real-time
    """)

    st.subheader("ข้อจำกัด")
    st.write("""
- ใช้ทรัพยากรสูง
    """)