import json
import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# streamlit-webrtc bersifat opsional: jika gagal diimpor, mode lain tetap berjalan.
try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
    WEBRTC_OK = True
except Exception:  # pragma: no cover
    WEBRTC_OK = False

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(page_title="Klasifikasi Ekspresi Wajah FER-2013", page_icon="🎭", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_fer2013_custom.keras")
LABEL_PATH = os.path.join(BASE_DIR, "class_names.json")
CASCADE_LOCAL = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

IMG_SIZE = 48
MAX_SIDE = 1280  # gambar yang lebih besar akan diperkecil agar proses cepat

# Warna kotak per ekspresi (BGR untuk OpenCV)
COLORS = {
    "angry": (60, 60, 230), "disgust": (60, 160, 60), "fear": (200, 90, 160),
    "happy": (60, 200, 255), "neutral": (200, 200, 200), "sad": (230, 130, 60),
    "surprise": (60, 170, 255),
}
EMOJI = {"angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
         "neutral": "😐", "sad": "😢", "surprise": "😲"}


# ==============================================================================
# 2. MUAT MODEL, LABEL, DAN DETEKTOR WAJAH
# ==============================================================================
@st.cache_resource(show_spinner="Memuat model...")
def load_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
        return None, None, None

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABEL_PATH, "r") as f:
        class_names = json.load(f)

    # Utamakan file XML bawaan OpenCV, cadangan: file XML di folder proyek.
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty() and os.path.exists(CASCADE_LOCAL):
        cascade = cv2.CascadeClassifier(CASCADE_LOCAL)
    if cascade.empty():
        cascade = None
    return model, class_names, cascade


model, class_names, face_cascade = load_assets()

if model is None:
    st.error("File `model_fer2013_custom.keras` atau `class_names.json` tidak ditemukan di folder proyek.")
    st.stop()
if face_cascade is None:
    st.error("Detektor wajah (Haar Cascade) gagal dimuat.")
    st.stop()


# ==============================================================================
# 3. FUNGSI INTI
# ==============================================================================
def preprocess_face(face_bgr):
    """Wajah BGR -> tensor (1, 48, 48, 1). Normalisasi 0-1 dilakukan oleh layer Rescaling di dalam model."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if face_bgr.ndim == 3 else face_bgr
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32)[None, :, :, None]


def predict_faces(face_list):
    """Prediksi beberapa wajah sekaligus (1 pemanggilan model). Mengembalikan array probabilitas (n, 7)."""
    if not face_list:
        return np.zeros((0, len(class_names)))
    batch = np.concatenate([preprocess_face(f) for f in face_list], axis=0)
    return model(batch, training=False).numpy()


def detect_faces(img_bgr, scale_factor=1.1, min_neighbors=5, min_size=48):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_size, min_size)
    )
    return [tuple(map(int, f)) for f in faces]


def draw_result(img_bgr, box, label, conf):
    x, y, w, h = box
    color = COLORS.get(label, (0, 255, 0))
    thick = max(2, img_bgr.shape[1] // 400)
    scale = max(0.5, img_bgr.shape[1] / 1000)
    cv2.rectangle(img_bgr, (x, y), (x + w, y + h), color, thick)
    text = f"{label} {conf * 100:.0f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    top = max(y - th - 10, 0)
    cv2.rectangle(img_bgr, (x, top), (x + tw + 8, top + th + 10), color, -1)
    cv2.putText(img_bgr, text, (x + 4, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick)


def resize_if_large(img_bgr):
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m > MAX_SIDE:
        s = MAX_SIDE / m
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img_bgr


def pil_to_bgr(pil_img):
    pil_img = ImageOps.exif_transpose(pil_img)  # perbaiki rotasi foto dari HP
    return resize_if_large(cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR))


def show_probabilities(scores):
    top3 = np.argsort(scores)[::-1][:3]
    for rank, idx in enumerate(top3, 1):
        name = class_names[idx]
        st.progress(float(scores[idx]), text=f"{rank}. {EMOJI.get(name, '')} {name.capitalize()}: {scores[idx] * 100:.1f}%")
    with st.expander("Lihat probabilitas semua kelas"):
        st.bar_chart({n.capitalize(): float(s) for n, s in zip(class_names, scores)})


def analyze_image(img_bgr, scale_factor, min_neighbors, min_size, allow_whole_image):
    """Deteksi wajah -> prediksi -> tampilkan gambar beranotasi dan hasil per wajah."""
    boxes = detect_faces(img_bgr, scale_factor, min_neighbors, min_size)
    whole_image_mode = False

    if not boxes:
        st.warning("Wajah tidak terdeteksi pada gambar ini.")
        if not allow_whole_image:
            st.caption("Coba gambar yang lebih jelas dan menghadap depan, kecilkan *Ukuran wajah minimum* di sidebar, "
                       "atau centang opsi 'Tetap prediksi seluruh gambar'.")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Gambar input", use_container_width=True)
            return
        h, w = img_bgr.shape[:2]
        boxes, whole_image_mode = [(0, 0, w, h)], True

    faces = [img_bgr[y:y + h, x:x + w] for (x, y, w, h) in boxes]
    probs = predict_faces(faces)

    annotated = img_bgr.copy()
    for box, p in zip(boxes, probs):
        i = int(np.argmax(p))
        draw_result(annotated, box, class_names[i], float(p[i]))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Hasil deteksi", use_container_width=True)
    with col2:
        if whole_image_mode:
            st.info("Seluruh gambar dipakai sebagai input karena wajah tidak terdeteksi. Hasil ini **kurang andal**.")
        else:
            st.success(f"Terdeteksi {len(boxes)} wajah.")
        for n, p in enumerate(probs, 1):
            i = int(np.argmax(p))
            st.markdown(f"### Wajah {n}: {EMOJI.get(class_names[i], '')} `{class_names[i].upper()}`")
            show_probabilities(p)
            if n < len(probs):
                st.divider()


# ==============================================================================
# 4. WEBCAM REAL-TIME (WebRTC)
# ==============================================================================
if WEBRTC_OK:
    class EmotionProcessor(VideoProcessorBase):
        PREDICT_EVERY = 3  # prediksi tiap 3 frame agar video tetap lancar

        def __init__(self):
            self.frame_idx = 0
            self.last = []  # [(label, conf), ...] sesuai urutan wajah terakhir

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            boxes = detect_faces(img, scale_factor=1.2, min_neighbors=5, min_size=60)

            need_predict = (self.frame_idx % self.PREDICT_EVERY == 0) or (len(boxes) != len(self.last))
            if boxes and need_predict:
                probs = predict_faces([img[y:y + h, x:x + w] for (x, y, w, h) in boxes])
                self.last = [(class_names[int(np.argmax(p))], float(np.max(p))) for p in probs]
            elif not boxes:
                self.last = []

            for box, (label, conf) in zip(boxes, self.last):
                draw_result(img, box, label, conf)

            self.frame_idx += 1
            return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==============================================================================
# 5. SIDEBAR
# ==============================================================================
st.title("🎭 Klasifikasi Ekspresi Wajah (FER-2013)")
st.caption("Proyek Akhir Kecerdasan Buatan — model Custom CNN (tanpa transfer learning)")

st.sidebar.success("✅ Model Custom CNN berhasil dimuat")
st.sidebar.header("⚙️ Pengaturan")

modes = ["Upload Gambar", "Kamera (Ambil Foto)"]
if WEBRTC_OK:
    modes.append("Webcam Real-Time")
mode = st.sidebar.radio("Pilih sumber input:", modes)

with st.sidebar.expander("Pengaturan deteksi wajah"):
    scale_factor = st.slider("Scale factor", 1.05, 1.5, 1.1, 0.05,
                             help="Lebih kecil = lebih teliti tetapi lebih lambat.")
    min_neighbors = st.slider("Min neighbors", 1, 10, 5,
                              help="Lebih besar = lebih sedikit salah deteksi, tetapi wajah samar bisa terlewat.")
    min_size = st.slider("Ukuran wajah minimum (px)", 24, 200, 48)
allow_whole = st.sidebar.checkbox("Tetap prediksi seluruh gambar jika wajah tidak terdeteksi", value=False)

with st.sidebar.expander("ℹ️ Tentang model & keterbatasan"):
    st.markdown(
        "- Dilatih pada **FER-2013** (citra grayscale 48×48, 7 ekspresi).\n"
        "- Akurasi pada FER-2013 berkisar 60–70%; dataset ini memiliki label yang berisik.\n"
        "- Kelas **disgust** dan **fear** paling sulit dikenali (data sedikit/ambigu).\n"
        "- Hasil terbaik: wajah menghadap depan, pencahayaan cukup, tidak tertutup."
    )

st.write("---")

# ==============================================================================
# 6. MODE-MODE INPUT
# ==============================================================================
if mode == "Upload Gambar":
    st.subheader("🖼️ Deteksi ekspresi dari file gambar")
    up = st.file_uploader("Pilih gambar (JPG / JPEG / PNG):", type=["jpg", "jpeg", "png"])
    if up is not None:
        try:
            img = pil_to_bgr(Image.open(up))
        except Exception:
            st.error("File tidak dapat dibaca sebagai gambar.")
        else:
            analyze_image(img, scale_factor, min_neighbors, min_size, allow_whole)

elif mode == "Kamera (Ambil Foto)":
    st.subheader("📸 Deteksi ekspresi dari foto kamera")
    st.caption("Mode ini memakai kamera browser dan tidak membutuhkan koneksi streaming, sehingga paling stabil.")
    shot = st.camera_input("Arahkan wajah ke kamera lalu ambil foto")
    if shot is not None:
        analyze_image(pil_to_bgr(Image.open(shot)), scale_factor, min_neighbors, min_size, allow_whole)

else:  # Webcam Real-Time
    st.subheader("📹 Deteksi ekspresi real-time via webcam")
    st.info("Tekan **START** lalu izinkan akses kamera. Jika video tidak muncul (jaringan tertentu memblokir WebRTC), "
            "gunakan mode **Kamera (Ambil Foto)**.")
    rtc_config = RTCConfiguration({"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
    ]})
    webrtc_streamer(
        key="fer-realtime",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=EmotionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
