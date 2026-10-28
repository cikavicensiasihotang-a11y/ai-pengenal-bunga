import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from pathlib import Path

# Konfigurasi halaman
st.set_page_config(
    page_title="AI Pengenal Bunga",
    page_icon="🌸",
    layout="centered"
)

# Lokasi model dan label di Google Drive
FOLDER = Path("/content/drive/MyDrive/AI_Pengenal_Bunga")
FILE_MODEL = FOLDER / "model_bunga_mobilenet.keras"
FILE_LABEL = FOLDER / "kelas_bunga_mobilenet.json"

st.title("🌸 AI Pengenal Bunga")
st.write(
    "Unggah foto atau gunakan kamera untuk mengenali "
    "satu dari 10 jenis bunga."
)

@st.cache_resource
def muat_model():
    model = tf.keras.models.load_model(
        FILE_MODEL,
        compile=False
    )
    with open(FILE_LABEL, "r", encoding="utf-8") as f:
        kelas = json.load(f)
    return model, kelas

try:
    model, kelas_bunga = muat_model()
    st.success("Model AI berhasil dimuat!")
except Exception as e:
    st.error("Model gagal dimuat. Periksa file model dan JSON.")
    st.exception(e)
    st.stop()

def prediksi_bunga(file_gambar):
    gambar = Image.open(file_gambar).convert("RGB")
    gambar_tampil = gambar.copy()
    gambar = gambar.resize((224, 224))

    array_gambar = np.array(gambar, dtype=np.float32) / 255.0
    array_gambar = np.expand_dims(array_gambar, axis=0)

    hasil = model.predict(array_gambar, verbose=0)[0]
    indeks = int(np.argmax(hasil))

    return gambar_tampil, kelas_bunga[indeks], float(hasil[indeks]) * 100

tab_upload, tab_kamera = st.tabs(["📁 Unggah Foto", "📷 Kamera"])

with tab_upload:
    file_upload = st.file_uploader(
        "Pilih foto bunga",
        type=["jpg", "jpeg", "png", "webp"],
        key="upload"
    )

    if file_upload is not None:
        gambar, nama, keyakinan = prediksi_bunga(file_upload)
        st.image(gambar, caption="Foto yang dipilih", use_container_width=True)
        st.subheader(f"🌼 Prediksi: {nama.replace('_', ' ').title()}")
        st.write(f"Tingkat kepercayaan: **{keyakinan:.2f}%**")

with tab_kamera:
    foto_kamera = st.camera_input("Ambil foto bunga")

    if foto_kamera is not None:
        gambar, nama, keyakinan = prediksi_bunga(foto_kamera)
        st.image(gambar, caption="Foto dari kamera", use_container_width=True)
        st.subheader(f"🌼 Prediksi: {nama.replace('_', ' ').title()}")
        st.write(f"Tingkat kepercayaan: **{keyakinan:.2f}%**")

st.divider()
st.caption("Proyek AI Pengenal Bunga — model klasifikasi 10 jenis bunga.")
