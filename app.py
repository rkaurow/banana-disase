from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.ai_response import FINAL_SYSTEM_PROMPT, generate_disease_response, get_ai_runtime_status

from src.inference import (
    DISEASE_INFO,
    load_artifacts,
    predict_image,
)

st.set_page_config(
    page_title="Banana Doctor AI",
    page_icon="🍌",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 225, 0, 0.15), transparent 26%),
                radial-gradient(circle at top right, rgba(34, 139, 34, 0.18), transparent 24%),
                linear-gradient(180deg, #07111a 0%, #0d1824 55%, #101522 100%);
            color: #edf3fb;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1624 0%, #121f30 100%);
            border-right: 1px solid rgba(152, 168, 186, 0.14);
        }
        .hero-card, .result-card, .metric-card {
            background: rgba(18, 27, 39, 0.92);
            border: 1px solid rgba(126, 145, 167, 0.18);
            border-radius: 22px;
            box-shadow: 0 18px 44px rgba(0, 0, 0, 0.28);
        }
        .hero-card { padding: 28px 30px; margin-bottom: 18px; }
        .hero-eyebrow { font-size: 0.8rem; letter-spacing: 0.14em; text-transform: uppercase; color: #ffeb3b; font-weight: 700; margin-bottom: 12px; }
        .hero-title { font-size: 2.5rem; line-height: 1.05; color: #f4f8fc; font-weight: 800; margin: 0; }
        .hero-text { color: #bcc9d8; margin-top: 12px; font-size: 1rem; }
        .result-card { padding: 24px; margin-top: 10px; }
        .result-label { color: #ffeb3b; text-transform: uppercase; letter-spacing: 0.12em; font-size: 0.78rem; font-weight: 700; }
        .result-title { color: #f4f8fc; font-size: 1.8rem; font-weight: 800; margin: 6px 0 10px; }
        .result-copy { color: #bcc9d8; }
        .metric-card { padding: 18px 20px; min-height: 110px; }
        .metric-label { color: #8ea2b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; }
        .metric-value { color: #f4f8fc; font-size: 1.6rem; font-weight: 800; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(labels: list[str], mode: str) -> None:
    label_count = len(labels)
    labels_text = ", ".join(labels) if labels else "Belum ada label aktif"
    st.markdown(
        f"""
        <div class="hero-card">
          <div class="hero-eyebrow">Computer Vision for Banana Leaves</div>
          <h1 class="hero-title">Banana Doctor AI</h1>
          <p class="hero-text">
            Deteksi dini penyakit pada daun pisang menggunakan AI. 
            Unggah foto daun pisang untuk mendapatkan diagnosa instan dan saran penanganan.
          </p>
          <p class="hero-text"><strong>Mode model:</strong> {mode}</p>
          <p class="hero-text"><strong>{label_count} kelas aktif:</strong> {labels_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(prediction: dict[str, object]) -> None:
    ai_response = prediction.get("ai_response")
    predicted_label = prediction["label"]
    confidence = float(prediction["confidence"])
    disease = DISEASE_INFO.get(
        predicted_label,
        {
            "status": predicted_label,
            "severity": "Perlu ditinjau",
            "info": "Informasi detail belum tersedia.",
            "treatment": "Lakukan pengecekan lapangan.",
        },
    )

    headline = ai_response["headline"] if ai_response else disease["status"]
    summary = ai_response["summary"] if ai_response and ai_response["summary"] else disease["info"]

    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-label">Hasil Diagnosa</div>
          <div class="result-title">{headline}</div>
          <p class="result-copy">{summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{confidence * 100:.2f}%</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Keparahan</div><div class="metric-value">{disease["severity"]}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Label</div><div class="metric-value">{predicted_label}</div></div>', unsafe_allow_html=True)

    if ai_response:
        st.write("**Saran Penanganan AI**")
        for action in ai_response.get("actions", []):
            st.write(f"- {action}")
    else:
        st.info(disease["treatment"])


def main() -> None:
    inject_styles()
    
    if "history" not in st.session_state:
        st.session_state["history"] = []

    artifacts, error_message = load_artifacts()

    if error_message:
        st.warning(error_message)
        st.info("💡 Pastikan model '.keras' dan 'labels.json' ada di folder artifacts.")
        # Kita tampilkan UI hero meskipun model belum ada agar user melihat scaffoldnya
        render_hero([], "N/A")
    else:
        render_hero(artifacts["all_labels"], str(artifacts["mode"]))

    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        uploaded_file = st.file_uploader("Upload foto daun pisang", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_container_width=True)

    with right_col:
        st.write("### Analisa")
        analyze_button = st.button("Analisa sekarang", type="primary", use_container_width=True, disabled=uploaded_file is None or artifacts is None)

        if uploaded_file and analyze_button:
            with st.spinner("Menganalisa..."):
                prediction = predict_image(artifacts, image)
                try:
                    prediction["ai_response"] = generate_disease_response(prediction)
                except:
                    prediction["ai_response"] = None
                
                # Simpan hasil prediksi dan reset chatbot setiap kali ada analisa baru
                st.session_state["current_prediction"] = prediction
                st.session_state.chat_messages = [
                    {"role": "assistant", "content": f"Halo! Hasil analisa menunjukkan **{prediction['label']}**. Ada yang ingin ditanyakan lebih lanjut mengenai hasil ini?"}
                ]
                
                render_result(prediction)
                st.session_state["history"].insert(0, {
                    "Waktu": datetime.now().strftime("%H:%M:%S"),
                    "Diagnosa": prediction["label"],
                    "Confidence": f"{prediction['confidence']*100:.1f}%"
                })

    if st.session_state["history"]:
        st.write("### Riwayat")
        st.table(pd.DataFrame(st.session_state["history"]))

    # Tampilkan chatbot hanya jika ada hasil prediksi aktif
    if "current_prediction" in st.session_state:
        st.write("---")
        st.write("### 🤖 Asisten Tanya Jawab (Chatbot)")
        
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("Ketik pertanyaan Anda di sini..."):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Mengetik..."):
                    from src.ai_response import chat_with_bot
                    pred = st.session_state["current_prediction"]
                    
                    # Tambahkan konteks hasil prediksi ke dalam prompt agar AI paham apa yang sedang dibahas
                    context_msg = {
                        "role": "system",
                        "content": f"Kamu adalah asisten pertanian ahli penyakit pisang. User sedang melihat hasil deteksi gambar daun pisangnya dengan hasil: {pred['label']} (Tingkat Keyakinan: {pred['confidence']*100:.2f}%). Berikan jawaban yang membantu, singkat, dan berhubungan dengan penyakit tersebut."
                    }
                    api_messages = [context_msg] + st.session_state.chat_messages
                    
                    response = chat_with_bot(api_messages)
                    if response:
                        st.write(response)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    else:
                        st.error("Gagal memanggil API Chatbot. Pastikan konfigurasi API key sudah benar atau layanan tersedia.")


if __name__ == "__main__":
    main()
