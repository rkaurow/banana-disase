# Panduan Belajar Ujian Skripsi - Banana Doctor AI

Dokumen ini disusun untuk persiapan presentasi dan tanya-jawab ujian skripsi berdasarkan implementasi nyata di codebase proyek Banana Doctor AI, yang kini menggunakan arsitektur two-stage: banana gate untuk validasi daun pisang, lalu ensemble 3 model untuk klasifikasi 8 kondisi daun pisang.

## 1. Elevator Pitch (30 detik)

Banana Doctor AI adalah sistem diagnosis penyakit daun pisang berbasis Computer Vision (Two-Stage Learning + Ensemble Learning) dan Large Language Model (LLM).
Sistem menerima gambar daun, menjalankan banana gate untuk memutuskan apakah gambar benar daun pisang, lalu melakukan klasifikasi penyakit menggunakan gabungan 3 model (Custom CNN, ResNet50, InceptionV3) dengan teknik *Weighted Soft Voting*. Terakhir, LLM menghasilkan rekomendasi penanganan yang spesifik dan praktis. Sistem ini tersedia di web dashboard FastAPI dan terintegrasi langsung dengan WhatsApp bot untuk kemudahan akses petani di lapangan.

## 2. Arsitektur Sistem End-to-End

Alur utama:

1. User mengirim foto dari Web atau WhatsApp.
2. Backend FastAPI menerima file gambar.
3. Modul inferensi menjalankan `banana_gate.keras`. Jika probabilitas `Banana Leaf` di bawah threshold, proses dihentikan dan sistem mengembalikan `Not Banana Leaf`.
4. Jika lolos gate, sistem memproses gambar dengan 3 model AI (Ensemble) dan melakukan *Weighted Soft Voting* untuk menentukan hasil akhir penyakit.
5. Sistem mengambil informasi statis dari database untuk penyakit terkait.
6. Jika prediksi adalah penyakit (bukan daun sehat), LLM membangkitkan rekomendasi tindakan medis pertanian.
7. Hasil dikembalikan ke frontend atau WhatsApp bot.

Komponen kunci:

1. Training two-stage: `train-collabs.ipynb`
2. Evaluasi model: `evaluate.py`
3. Inferensi banana gate + Soft Voting: `src/inference.py`
4. AI response + chat: `src/ai_response.py`
5. API server: `api.py`
6. Frontend web: `frontend/index.html` + `frontend/js/app_v2.js`
7. WhatsApp bot: `whatsapp-bot/bot.js`

## 3. Mapping File Penting untuk Presentasi

1. Backend API: `api.py`
2. Inference core: `src/inference.py`
3. LLM integration: `src/ai_response.py`
4. Training notebook: `train-collabs.ipynb`
5. Evaluation script: `evaluate.py`
6. Frontend entry: `frontend/index.html`
7. Frontend logic: `frontend/js/app_v2.js`
8. WhatsApp bot: `whatsapp-bot/bot.js`
9. Konfigurasi banana gate: `artifacts/banana_gate_config.json`
10. Konfigurasi Ensemble: `artifacts/ensemble_config.json`
11. Daftar Label penyakit: `artifacts/labels.json`

## 4. Konsep Machine Learning yang Harus Siap Dijelaskan

### 4.1 Dataset dan 8 Kelas Penyakit/Healthy

Disease classifier dilatih menggunakan dataset Kaggle berukuran besar yang mencakup 8 kelas daun pisang:

1. Black Sigatoka Disease (Sigatoka Hitam)
2. Bract Mosaic Virus Disease (Virus Mosaik Seludang)
3. Cordana Disease (Bercak Daun Cordana)
4. Healthy Leaf (Daun Sehat)
5. Insect Pest Disease (Hama Serangga)
6. Moko Disease (Layu Bakteri)
7. Panama Disease (Layu Fusarium)
8. Yellow Sigatoka Disease (Sigatoka Kuning)

Banana gate dilatih sebagai binary classifier:
1. `Banana Leaf`: gabungan semua 8 kelas daun pisang.
2. `Not Banana Leaf`: daun non-pisang, objek non-daun, scene indoor, natural images, PlantDoc, dan `hard_negatives/`.

### 4.2 Arsitektur Ensemble Learning (Penting untuk Skripsi)

Sistem menggunakan teknik **Ensemble Learning** dengan metode **Weighted Soft Voting**.
Artinya, kita tidak hanya mengandalkan 1 model, melainkan 3 model sekaligus:
1. **Custom CNN**: Arsitektur ringan buatan sendiri yang dilatih dari awal.
2. **ResNet50**: Pre-trained model (Transfer Learning) yang andal dalam mengekstrak fitur residual.
3. **InceptionV3**: Pre-trained model yang andal mengekstrak fitur pada berbagai skala (inception modules).

**Kenapa Ensemble?**
Argumen akademik: Menggabungkan beberapa model dengan arsitektur berbeda dapat mengurangi varians (overfitting), menutupi kelemahan satu model dengan kelebihan model lain, dan menghasilkan prediksi akhir yang jauh lebih stabil (robust) dibanding model tunggal.

### 4.3 Weighted Soft Voting

Model menggunakan rata-rata terbobot (Weighted Soft Voting). Jika akurasi model ResNet50 lebih rendah (misal 65%) dibanding CNN dan InceptionV3 (misal 90%), sistem akan **memberikan bobot lebih kecil** pada keputusan ResNet50 dan **bobot lebih besar** pada CNN/Inception. Ini mencegah model yang lemah merusak hasil akhir.

### 4.4 Banana Gate

Sebelum gambar dianalisa penyakitnya, sistem menggunakan `banana_gate.keras` untuk membedakan `Banana Leaf` vs `Not Banana Leaf`. Threshold disimpan di `banana_gate_config.json` dan dipilih dari validation set dengan target recall tinggi untuk kelas negatif maupun daun pisang. Heuristic ImageNet/color/skin hanya dipertahankan sebagai fallback legacy ketika artifact two-stage belum tersedia.

## 5. Detail Training Notebook (`train-collabs.ipynb`)

Yang harus Anda pahami:

1. `banana_gate.keras` dilatih lebih dulu sebagai binary classifier, memakai transfer learning dan fine-tuning layer atas.
2. Disease ensemble dilatih setelahnya hanya pada 8 kelas daun pisang.
3. Data Augmentation: Rotation, Shift, Shear, Zoom, Horizontal Flip.
3. **Callback MLOps**: 
    - `ModelCheckpoint`: Menyimpan model hanya saat akurasi validasi terbaik (`best_cnn.keras`, dsb).
    - `EarlyStopping`: Berhenti training jika tidak ada peningkatan (mencegah overfitting).
    - `ReduceLROnPlateau`: Menurunkan learning rate jika akurasi *nyangkut* (plateau).
4. Menyimpan konfigurasi ke `artifacts/banana_gate_config.json` dan `artifacts/ensemble_config.json` (termasuk bobot akurasi masing-masing model).

## 6. Detail Inferensi dan Alasan Desain (`src/inference.py`)

1. Load `artifacts/banana_gate.keras`, `artifacts/banana_gate_config.json`, `artifacts/ensemble_config.json`, dan ketiga model penyakit.
2. **Preprocessing**: Resize (224x224), konversi RGB, normalisasi (0-1).
3. **Banana Gate**: Jika probabilitas `Banana Leaf` di bawah threshold, kembalikan label `Not Banana Leaf`.
4. **Weighted Voting**: Jika lolos gate, hitung probabilitas dari 3 model penyakit, kalikan dengan bobot akurasi kuadrat masing-masing model, lalu ambil indeks dengan rata-rata tertinggi.

## 7. Integrasi LLM (`src/ai_response.py`)

1. LLM (Large Language Model) memberikan jawaban *human-like* berbahasa Indonesia.
2. **Prompt Engineering**: Prompt dirancang memaksa LLM mengembalikan format JSON (`headline, summary, meaning, actions, prevention, warning`).
3. Memaksa AI memberikan **Merek Fungisida/Pestisida spesifik** (Amistar Top, Dithane, Nordox, dll) agar aplikasi benar-benar praktis untuk petani Indonesia.
4. AI tidak akan terpanggil jika prediksi model adalah "Healthy Leaf" (Daun Sehat) untuk menghemat biaya (Token/API cost) dan mempercepat waktu tunggu.

## 8. Frontend Web (Dashboard)

1. Tampilan UI/UX modern berbasis Vanilla CSS, mendukung mode gelap (Dark Mode).
2. Memiliki *Katalog Referensi Penyakit* yang sudah di-update dengan ke-7 penyakit sesuai dataset.
3. Menampilkan tingkat keyakinan (Confidence Score) secara dinamis menggunakan SVG Circular Progress bar.
4. Fitur chatbot terintegrasi untuk tanya-jawab lanjutan mengenai penyakit spesifik.

## 9. WhatsApp Bot (`whatsapp-bot/bot.js`)

Nilai Jual Utama Skripsi:
1. **Aksesibilitas Tinggi**: Petani di desa sering kali lebih nyaman memakai WhatsApp dibanding website.
2. **Penanganan OOD Spesifik**: Jika user upload screenshot HP, WA Bot secara otomatis menolak dan memberikan saran cara pengambilan foto daun yang benar.
3. **Session Memory**: Jika pengguna bertanya lanjutan (misal: "Berapa dosis Nordox yang pas?"), bot WA akan mengingat konteks penyakit yang baru saja didiagnosis.

## 10. Rumus Penting untuk Ujian Skripsi

Bagian ini dibuat agar Anda siap saat dosen penguji bertanya matematika di baliknya.

### 10.1 Weighted Soft Voting (Inti Ensemble)

Untuk menghitung probabilitas final ( $P_f$ ) dari kelas ke-$k$:

$$
P_f(k) = \sum_{i=1}^{N} W_i \cdot P_i(k)
$$

Di mana:
- $N$ adalah jumlah model (ada 3: CNN, ResNet, Inception).
- $P_i(k)$ adalah prediksi probabilitas dari model ke-$i$ untuk kelas $k$ (hasil dari *softmax* layer model tersebut).
- $W_i$ adalah bobot dari model ke-$i$.

**Cara Menghitung Bobot ($W_i$) di Sistem Ini:**
Kita menggunakan *akurasi model yang dikuadratkan* untuk memberikan hukuman pada model yang jelek (misal ResNet50):

$$
W_i = \frac{Accuracy_i^2}{\sum_{j=1}^{N} Accuracy_j^2}
$$

### 10.2 Metrik Evaluasi Klasifikasi

1. **Accuracy**: Total benar dibagi keseluruhan data.
$$ Accuracy = \frac{TP + TN}{TP + TN + FP + FN} $$
2. **Precision**: Seberapa akurat model saat memprediksi suatu penyakit.
$$ Precision = \frac{TP}{TP + FP} $$
3. **Recall**: Dari seluruh daun yang *benar-benar sakit*, seberapa banyak yang berhasil ditangkap oleh model.
$$ Recall = \frac{TP}{TP + FN} $$
4. **F1-Score**: Rata-rata harmonik Precision dan Recall (Sangat penting saat data imbalance).
$$ F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall} $$

## 11. Mermaid Chart untuk Slide Skripsi

### 11.1 Chart Arsitektur End-to-End Two-Stage

```mermaid
flowchart LR
    A[User Web / WA] --> B[FastAPI /api/predict]
    B --> C[Preprocessing RGB & Resize]
    C --> D[Banana Gate]
    D -->|Not Banana Leaf| E[Tolak: Not Banana Leaf]
    D -->|Banana Leaf| F[Preprocessing Disease Classifier]
    
    F --> F1[Custom CNN]
    F --> F2[ResNet50]
    F --> F3[InceptionV3]
    
    F1 --> G[Weighted Soft Voting]
    F2 --> G
    F3 --> G
    
    G --> H[Prediksi Final Label & Confidence]
    H --> I{Apakah Sehat?}
    I -->|Ya| J[Kembalikan Hasil Sehat]
    I -->|Tidak| K[LLM: Generate Rekomendasi]
    K --> L[Format JSON API]
    J --> L
    L --> M[Kirim ke Web / Bot WA]
```

## 12. Pertanyaan Dosen dan Jawaban Cerdas (Cheat-Sheet)

1. **Dosen:** "Kenapa repot-repot pakai 3 model (Ensemble) padahal Custom CNN saja akurasinya 90%?"
   **Jawaban:** "Untuk *Robustness* (ketahanan) pak/bu. Di dunia nyata (lapangan), kondisi pencahayaan sangat bervariasi. Walaupun CNN bagus di data validasi, dia punya *blind-spots*. Dengan menggabungkannya bersama InceptionV3 dan ResNet50, kita menutupi blind-spots tersebut, sehingga meminimalkan salah prediksi saat dipakai petani secara riil."

2. **Dosen:** "Bagaimana kalau ada model di Ensemble yang akurasinya jelek banget, bukannya malah merusak hasil?"
   **Jawaban:** "Itu sudah saya atasi menggunakan *Weighted Soft Voting*. Model yang performanya rendah (seperti ResNet50) diberi bobot yang jauh lebih kecil berdasarkan kuadrat akurasinya, sehingga suaranya tidak akan 'menyeret' turun prediksi benar dari model lain yang bagus."

3. **Dosen:** "Kenapa pakai AI/ChatGPT lagi padahal sudah ada klasifikasi gambar?"
   **Jawaban:** "Klasifikasi hanya memberi tahu *'apa penyakitnya'*. LLM digunakan untuk *'apa yang harus dilakukan petani'*. Integrasi LLM memastikan rekomendasinya spesifik, berbahasa Indonesia, menyebut merek fungisida/pestisida lokal yang ada di pasaran, serta sangat interaktif lewat WhatsApp."

4. **Dosen:** "Bagaimana kalau petani iseng foto muka atau tangannya, apa AI akan menebak itu penyakit pisang?"
   **Jawaban:** "Tidak pak/bu. Sistem memiliki layer deteksi Out-Of-Distribution (OOD) di awal. Jika sistem mendeteksi wajah, hewan, screenshot layar HP, atau teks, proses akan langsung di-blok dan muncul peringatan 'Bukan Daun Pisang' beserta tips fotografi yang benar."
