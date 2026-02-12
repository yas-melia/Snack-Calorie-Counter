import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Snack Calories
SNACK_KCAL = {
    "TOP": 160, "Maxx": 160, "Cadbury": 260, "Delfi": 228,
    "BAR BAR": 90, "SilverQueen": 330, "VAN HOUTEN": 222
}

# Box Colours
CLASS_COLORS = {
    0: (255, 0, 0),    # TOP - Red
    1: (0, 255, 255),  # Maxx - Cyan
    2: (128, 0, 128),  # Cadbury - Purple
    3: (255, 165, 0),  # Delfi - Orange
    4: (255, 255, 0),  # BAR BAR - Yellow
    5: (0, 0, 255),    # SilverQueen - Blue
    6: (139, 69, 19),  # VAN HOUTEN - Brown
}

# Load Model
@st.cache_resource
def load_model():
    try:
        return YOLO('model/best.pt') # Make sure the path is correct, in this case my "best.pt" file is in this directory
    except:
        st.warning("Model is not found. We're using YOLO default model.")
        return YOLO('yolo11n.pt')

model = load_model()

def detect_objects(image):
    results = model(image, conf=0.5, iou=0.45)
    total_calories = 0
    img_draw = np.array(image)
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            name = model.names[cls]
            color = CLASS_COLORS.get(cls, (0, 255, 0))
            
            total_calories += SNACK_KCAL.get(name, 0)
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{name} {conf:.2f}"
            
            # Box & Label
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 3)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_draw, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img_draw, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Calories in Total Overlay
    cv2.rectangle(img_draw, (0, 0), (280, 60), (0, 0, 0), -1) 
    cv2.putText(img_draw, f"TOTAL: {total_calories} kcal", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    
    return img_draw

# Streamlit UI
st.set_page_config(page_title="Snack Calorie Counter", layout="wide")
st.title("🍫 Snack Calorie Counter")

# Sidebar
st.sidebar.title("Settings")
input_mode = st.sidebar.radio("Choose an Input Mode:", ("Snapshot", "Media Upload"))

if input_mode == "Snapshot":
    st.subheader("Snapshot Mode")
    
    # Layouting for better looks
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        img_file = st.camera_input("Show your snack(s) to the camera")

        if img_file:
            image = Image.open(img_file).convert("RGB")
            
            with st.spinner('Analyzing your photo...'):
                # Detection process
                output_image = detect_objects(image)
                
                # Result
                st.write("### 🔍 Result")
                st.image(output_image, use_container_width=True)

elif input_mode == "Media Upload":
    st.subheader("Media Upload Mode")
    uploaded_file = st.file_uploader("Choose a file...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.write("Analyzing your photo...")
        
        output_image = detect_objects(image)
        st.image(output_image, caption="Result", use_container_width=True)