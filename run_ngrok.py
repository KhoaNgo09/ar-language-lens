from pyngrok import ngrok
import subprocess
import time

# --- Bước 1: Mở tunnel ngrok cho port 8501 ---
print("🚀 Khởi động ngrok tunnel...")
public_url = ngrok.connect(8501).public_url
print(f"✅ Link chia sẻ Streamlit: {public_url}")

# --- Bước 2: Chạy Streamlit app ---
print("📸 Đang chạy Streamlit app...")
process = subprocess.Popen(["streamlit", "run", "app.py"])

# --- Bước 3: Giữ chương trình chạy ---
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("🛑 Dừng ngrok & Streamlit...")
    process.terminate()
    ngrok.disconnect(public_url)
