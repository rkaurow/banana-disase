# 🍌 Banana Doctor AI

Banana Doctor AI adalah aplikasi deteksi penyakit daun pisang berbasis **Computer Vision (MobileNetV2)** dan **AI Asisten (LLM)** dengan arsitektur **FastAPI** dan **Modern Dashboard**. Aplikasi ini juga dilengkapi dengan integrasi **WhatsApp Bot** yang memungkinkan para petani melakukan diagnosa secara langsung melalui WhatsApp.

## ✨ Fitur Utama

- **Deteksi Penyakit Daun Pisang Secara Real-time**: Menggunakan model MobileNetV2 untuk mengenali penyakit seperti *Cordana*, *Panama Disease*, *Yellow and Black Sigatoka*, dan daun sehat (*Healthy*).
- **Dashboard Modern Premium**: UI/UX menggunakan desain *Glassmorphism* dan *Dark Mode* yang memukau dan ringan.
- **Asisten AI Terintegrasi**: Menggunakan LLM (seperti GPT-4o-mini melalui Sumopod AI) untuk memberikan rekomendasi perawatan, tindakan pencegahan, dan produk penanganan yang spesifik.
- **WhatsApp Bot**: Fitur bot WhatsApp cerdas yang memungkinkan pengguna untuk mengunggah foto daun langsung ke WhatsApp, menerima hasil diagnosa secara instan, dan melakukan sesi tanya jawab (chat) dengan asisten AI seputar penyakit tersebut.
- **Pelatihan Model Dua Tahap (Two-Stage Training)**: Kemampuan melatih model menjadi sistem biner (Sehat vs Sakit) dan spesialis penyakit untuk akurasi yang lebih tinggi.

---

## 🛠 Persiapan & Instalasi

### 1. Kebutuhan Sistem & Instalasi
Pastikan Anda sudah menginstal Node.js (untuk bot WhatsApp) dan Python 3.8+ (untuk backend API).

```bash
# Clone repository ini (jika belum)
git clone https://github.com/rkaurow/banana-disase.git
cd banana-disase

# Install dependensi Python (Backend FastAPI)
pip install -r requirements.txt

# Install dependensi Node.js (WhatsApp Bot)
cd whatsapp-bot
npm install
cd ..
```

### 2. Konfigurasi AI (LLM & Chatbot)
Aplikasi ini menggunakan API LLM cerdas untuk menghasilkan deskripsi dan menangani sesi *chat*.
1. Buat file `.env` di root direktori utama.
2. Tambahkan API Key Anda:
   ```env
   SUMOPOD_API_KEY=isi_api_key_disini
   ```
   *Catatan: Endpoint API secara default mengarah ke `https://api.sumopod.com/v1`. Jika menggunakan API OpenAI, sesuaikan konfigurasi `base_url` di dalam file `src/ai_response.py`.*

---

## 🚀 Cara Menjalankan Layanan

### 1. Menjalankan Server Utama (FastAPI Backend & Web Dashboard)
Server backend menangani pengolahan gambar (ML) dan interaksi dengan LLM. Web dashboard di-*serve* langsung dari backend ini.
```bash
uvicorn api:app --reload
```
- **Web Dashboard Utama**: [http://localhost:8000](http://localhost:8000)
- **Dokumentasi API (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Menjalankan WhatsApp Bot (Opsional/Pendamping)
Bot WhatsApp ini berjalan sebagai service mandiri (Node.js) yang memanggil Backend FastAPI Anda. Pastikan **Backend FastAPI sudah berjalan** sebelum menyalakan bot.

Buka tab terminal baru:
```bash
cd whatsapp-bot
node bot.js
```
- Perhatikan terminal, akan muncul sebuah **QR Code**.
- Buka aplikasi WhatsApp di HP Anda -> **Perangkat Tertaut** -> **Tautkan Perangkat** lalu scan QR Code tersebut.
- Setelah berhasil, bot siap digunakan. Kirimkan pesan atau foto daun pisang ke nomor bot tersebut untuk mulai!

---

## 📂 Struktur Proyek

- `api.py`: Script utama backend **FastAPI** yang menyediakan endpoint API (`/api/predict`, `/api/chat`) dan Web Dashboard statis.
- `frontend/`: Direktori antarmuka pengguna (HTML, CSS Glassmorphism, JS Logic).
- `whatsapp-bot/`: Service Bot WhatsApp menggunakan `@whiskeysockets/baileys`.
- `src/`: Modul *core* (Pemuatan Model ML, Prediksi, Integrasi LLM AI).
- `train.py` / `train_two_stage.py`: Script untuk melakukan proses pelatihan (*training*) ulang dari dataset lokal.
- `evaluate.py`: Script untuk evaluasi model (Confusion Matrix & Classification Report).
- `guides.md`: Panduan lengkap penggunaan teknis dan pengembangan.

---

## 🧠 Training & Evaluasi (Opsional)
Jika Anda memiliki dataset yang diletakkan di `datasets/`, Anda dapat melatih ulang model:

1. **Single-Stage**: `python3 train.py` (satu model langsung untuk 4 kelas).
2. **Two-Stage**: `python3 train_two_stage.py` (model *Sehat/Sakit* lalu model spesialis penyakit).
3. **Evaluasi**: `python3 evaluate.py` (menyimpan model `.keras` ke dalam `artifacts/`).

---

## 📈 Tips untuk Hasil Maksimal
- **Kualitas Gambar**: Gunakan pencahayaan alami yang baik saat memotret daun. Pastikan fokus objek berada di area daun yang terinfeksi dan terhindar dari *blur*.
- **Refresh Dashboard**: Jika mendapati sedikit anomali pada UI/UX web browser, lakukan *Hard Refresh* dengan menekan `Ctrl + Shift + R`.
- **Bot Chat**: Ketik **"reset"** di WhatsApp Bot untuk menghapus riwayat obrolan AI jika Anda ingin berkonsultasi mengenai foto tanaman yang baru.
