
import cv2
import cvzone
import math
import streamlit as st
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# --- Hàm hỗ trợ hiển thị tiếng Việt ---
def draw_vietnamese_text(img1, text, position, font_size=24, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- Streamlit UI ---
st.set_page_config(page_title="AR Language Lens", page_icon="📸", layout="centered")
st.title("📷 AR Language Lens - YOLOv8")
st.write("Nhận diện vật thể và hiển thị tên tiếng Việt 🌏")

# --- Load model YOLO ---
import os
model_path = "yolov8m.pt"

# Nếu file chưa tồn tại, tải lại model từ Ultralytics
if not os.path.exists(model_path):
    from ultralytics import YOLO
    model = YOLO('yolov8m.pt')  # tự tải về từ hub
else:
    model = YOLO(model_path)

# Danh sách lớp tiếng Việt
classNames = [
    "Person - Con người", "Bicycle - Xe đạp", "Car - Ô tô", "Motorbike - Xe máy", "Aeroplane - Máy bay",
    "Bus - Xe buýt", "Train - Tàu hỏa", "Truck - Xe tải", "Boat - Thuyền",
    "Traffic Light - Đèn giao thông", "Fire Hydrant - Trụ nước cứu hỏa", "Stop Sign - Biển dừng",
    "Parking Meter - Đồng hồ đỗ xe", "Bench - Ghế dài", "Bird - Chim", "Cat - Mèo",
    "Dog - Chó", "Horse - Ngựa", "Sheep - Cừu", "Cow - Bò", "Elephant - Voi", "Bear - Gấu",
    "Zebra - Ngựa vằn", "Giraffe - Hươu cao cổ", "Backpack - Ba lô", "Umbrella - Ô/Dù",
    "Handbag - Túi xách", "Tie - Cà vạt", "Suitcase - Vali", "Frisbee - Đĩa ném",
    "Skis - Ván trượt tuyết", "Snowboard - Ván trượt tuyết (Một tấm)", "Sports Ball - Bóng thể thao",
    "Kite - Diều", "Baseball Bat - Gậy bóng chày", "Baseball Glove - Găng bóng chày",
    "Skateboard - Ván trượt", "Surfboard - Ván lướt sóng", "Tennis Racket - Vợt Tennis",
    "Bottle - Chai", "Wine Glass - Ly rượu", "Cup - Cốc", "Fork - Nĩa", "Knife - Dao",
    "Spoon - Thìa", "Bowl - Bát", "Banana - Chuối", "Apple - Táo", "Sandwich - Bánh Sandwich",
    "Orange - Cam", "Broccoli - Bông cải xanh", "Carrot - Cà rốt", "Hot Dog - Xúc xích kẹp bánh mì",
    "Pizza - Bánh Pizza", "Donut - Bánh Donut", "Cake - Bánh kem", "Chair - Ghế",
    "Sofa - Ghế Sô Pha", "Potted Plant - Cây cảnh", "Bed - Giường", "Dining Table - Bàn ăn",
    "Toilet - Bồn cầu", "TV Monitor - Tivi/Màn hình", "Laptop - Máy tính xách tay",
    "Mouse - Chuột máy tính", "Remote - Điều khiển", "Keyboard - Bàn phím", "Cell Phone - Điện thoại di động",
    "Microwave - Lò vi sóng", "Oven - Lò nướng", "Toaster - Máy nướng bánh mì", "Sink - Bồn rửa",
    "Refrigerator - Tủ lạnh", "Book - Sách", "Clock - Đồng hồ", "Vase - Bình hoa",
    "Scissors - Kéo", "Teddy Bear - Gấu bông", "Hair Drier - Máy sấy tóc", "Toothbrush - Bàn chải đánh răng"
]

# --- Chọn chế độ ---
mode = st.radio("Chọn chế độ:", ["🖼 Nhận diện ảnh", "📹 Nhận diện bằng webcam"])

# --- Xử lý ảnh upload ---
if mode == "🖼 Nhận diện ảnh":
    run = st.checkbox("Bắt đầu nhận diện")
    uploaded_file = st.file_uploader("📁 Tải ảnh lên để nhận diện", type=["jpg", "jpeg", "png"])
    FRAME_WINDOW = st.empty()

    if uploaded_file is not None and run:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                if conf < 0.5:
                    continue
                cls = int(box.cls[0])
                label = f"{classNames[cls]} {conf:.2f}"
                img = draw_vietnamese_text(img, label, (x1, y1 - 25), font_size=24, color=(255, 0, 255))

        FRAME_WINDOW.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

    elif not run:
        st.info("👆 Hãy chọn ảnh và bật 'Bắt đầu nhận diện' để chạy mô hình.")

# --- Xử lý webcam ---
elif mode == "📹 Nhận diện bằng webcam":

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.model = model

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = self.model(img, stream=True)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    if conf < 0.5:
                        continue
                    cls = int(box.cls[0])
                    label = f"{classNames[cls]} {conf:.2f}"
                    img = draw_vietnamese_text(img, label, (x1, y1 - 25), font_size=22, color=(255, 0, 255))
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
    st.info("📸 Cho phép quyền truy cập webcam khi trình duyệt hỏi để bắt đầu nhận diện.")



