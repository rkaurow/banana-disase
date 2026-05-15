# 🍌 Banana Doctor AI - Documentation Guide

Aplikasi deteksi penyakit daun pisang berbasis Computer Vision (MobileNetV2) dan AI Asisten (LLM) dengan arsitektur **FastAPI** dan **Modern Dashboard**.

## 🛠 Persiapan (Prerequisites)

### 1. Instalasi Library
Pastikan Anda sudah menginstal semua library yang dibutuhkan di environment Anda:
```bash
pip install -r requirements.txt
```

### 2. Struktur Dataset
Jika ingin melatih ulang model, pastikan dataset diletakkan di folder `datasets/` dengan struktur berikut:
```text
datasets/
├── Cordana/
├── Healthy/
├── Panama Disease/
└── Yellow and Black Sigatoka/
```

---

## 🚀 Cara Menjalankan Aplikasi

Aplikasi ini menggunakan **FastAPI** sebagai server backend dan antarmuka web statis Vanilla HTML/JS.

### 1. Menjalankan Server Utama
Untuk menjalankan server API dan Web Dashboard:
```bash
uvicorn api:app --reload
```

### 2. Mengakses Layanan
Setelah server berjalan, Anda dapat mengakses layanan melalui browser:
- **Web Dashboard Utama**: `http://localhost:8000`
- **Dokumentasi API (Swagger UI)**: `http://localhost:8000/docs`

---

## 🤖 Konfigurasi AI (LLM & Chatbot)

Aplikasi ini menggunakan model cerdas **gpt-4o-mini** untuk memberikan minimal 5 saran penanganan secara otomatis dan fitur Chat Asisten Pribadi.
1. Buat file `.env` di root direktori.
2. Tambahkan API Key dari Sumopod (atau OpenAI) Anda:
   ```text
   SUMOPOD_API_KEY=isi_api_key_disini
   ```
   *Note: Endpoint API mengarah ke base_url custom Sumopod (`https://api.sumopod.com/v1`). Jika Anda menggunakan OpenAI standar, Anda bisa menghapus `base_url` di dalam file `src/ai_response.py`.*

---

## 🧠 Training & Evaluasi Model

### 1. Melatih Model (Training)
Ada dua pilihan metode training (*Training Script*):

*   **Mode Single-Stage (Standar)**: Melatih satu model untuk 4 kelas sekaligus.
    ```bash
    python3 train.py
    ```
*   **Mode Two-Stage (Advanced)**: Melatih model biner (Sehat vs Sakit) dan model spesialis penyakit.
    ```bash
    python3 train_two_stage.py
    ```

### 2. Evaluasi Model
Untuk melihat Confusion Matrix dan Classification Report:
```bash
python3 evaluate.py
```
Hasil evaluasi dan *file model* (`.keras`) disimpan di folder `artifacts/`.

---

## 📂 Struktur Proyek Terkini

*   `api.py`: Script utama backend **FastAPI** (Menggantikan Streamlit). Berisi endpoint `/api/predict` dan `/api/chat`.
*   `frontend/`: Direktori antarmuka pengguna *(UI)*.
    *   `index.html`: Struktur utama Dashboard Premium.
    *   `css/styles_v2.css`: Desain estetika *Glassmorphism* dan *Dark Mode*.
    *   `js/app_v2.js`: Logika AJAX, *Chatbot*, dan interaksi visual grafis.
*   `src/inference.py`: Modul untuk memuat model Keras dan melakukan prediksi.
*   `src/ai_response.py`: Modul asisten GPT (prompt generator dan chat).
*   `train.py` & `train_two_stage.py`: Script pipeline pelatihan model.
*   `app.py`: Versi Streamlit lama (Opsional / Deprecated).

---

## 📈 Tips untuk Akurasi Tinggi
- Gunakan pencahayaan yang baik saat memotret daun.
- Pastikan objek utama adalah daun pisang (bukan latar belakang).
- Jika terjadi UI *glitch* di browser Anda, lakukan **Hard Refresh** (`Ctrl + Shift + R`).
