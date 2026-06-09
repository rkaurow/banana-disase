from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

# === Model Paths ===
ENSEMBLE_CONFIG_PATH = Path("artifacts/ensemble_config.json")
LABELS_PATH = Path("artifacts/labels.json")
# Fallback single model (backward compat)
SINGLE_MODEL_PATH = Path("artifacts/banana_disease_model.keras")
ARTIFACTS_PATH = Path("artifacts")
IMAGE_SIZE = 224

# Label khusus untuk gambar yang bukan daun pisang (mis. tangan, wajah, objek random).
NOT_BANANA_LABEL = "Not Banana Leaf"

# Ambang batas untuk Out-Of-Distribution detection.
# Strategi: BLOKIR hanya jika ada sinyal kuat "jelas bukan tumbuhan" (tangan, wajah, kendaraan, dll).
# Pendekatan positif (cari bukti IS plant) terlalu sering false-positive pada daun pisang
# yang difoto dari sudut tidak biasa atau dengan background tanah.
#
# OOD_NON_PLANT_BLOCK: jika total skor kelas "jelas bukan tumbuhan" >= nilai ini -> blokir.
# OOD_MIN_CONFIDENCE: jika model pisang sangat tidak yakin DAN skor non-plant cukup tinggi -> blokir.
OOD_NON_PLANT_BLOCK = 0.60   # Harus sangat yakin bukan tumbuhan sebelum blokir
OOD_MIN_CONFIDENCE  = 0.30   # Confidence minimum model pisang (pelengkap, bukan penentu utama)

# Kata kunci kelas ImageNet yang JELAS bukan tumbuhan/alam.
# Gambar dengan top-K ImageNet didominasi kelas ini -> blokir.
_NON_PLANT_KEYWORDS = (
    # Manusia & tubuh
    "hand", "face", "head", "neck", "arm", "leg", "foot", "finger", "thumb",
    "person", "people", "man", "woman", "boy", "girl", "child", "baby",
    "mask", "stocking", "sock", "shoe", "sneaker", "sandal", "boot",
    # Kendaraan
    "car", "truck", "bus", "bicycle", "motorcycle", "airplane", "ship", "boat",
    "van", "jeep", "ambulance", "minivan", "taxicab", "limousine",
    # Elektronik & peralatan
    "phone", "mobile", "laptop", "computer", "keyboard", "mouse", "monitor",
    "television", "remote", "camera", "refrigerator", "microwave", "toaster",
    # Hewan (bukan relevan untuk diagnosis daun)
    "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", "monkey",
    "snake", "lizard", "frog", "spider", "insect", "bee", "ant",
    # Makanan olahan / non-natural
    "pizza", "burger", "hot dog", "sandwich", "ice cream", "cake", "bread",
    "noodle", "sushi", "taco", "burrito",
    # Furnitur & bangunan
    "chair", "table", "desk", "sofa", "bed", "door", "window", "wall", "floor",
    "building", "house", "tower", "bridge",
)

_ood_backbone: tf.keras.Model | None = None
_ood_failed = False


def _get_ood_backbone() -> tf.keras.Model | None:
    """MobileNetV2 ImageNet (with top) untuk verifikasi 'gambar ini tumbuhan/daun atau bukan'."""
    global _ood_backbone, _ood_failed
    if _ood_backbone is not None or _ood_failed:
        return _ood_backbone
    try:
        _ood_backbone = tf.keras.applications.MobileNetV2(weights="imagenet")
    except Exception as exc:  # pragma: no cover - hanya runtime
        print(f"[inference] OOD backbone gagal dimuat: {exc}. OOD-check dilewati.")
        _ood_failed = True
        _ood_backbone = None
    return _ood_backbone


def _imagenet_non_plant_score(image: Image.Image) -> float | None:
    """Total probabilitas top-20 ImageNet yang termasuk kelas 'jelas bukan tumbuhan'.
    Tinggi (>0.60) berarti gambar hampir pasti bukan daun/tumbuhan.
    Mengembalikan None jika backbone gagal dimuat."""
    model = _get_ood_backbone()
    if model is None:
        return None
    arr = np.asarray(image.convert("RGB").resize((224, 224)), dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=20)[0]
    score = 0.0
    for _, name, prob in decoded:
        n = name.lower()
        if any(k in n for k in _NON_PLANT_KEYWORDS):
            score += float(prob)
    return score


def _not_banana_payload(reason: str, non_plant_score: float | None, confidence: float | None) -> dict[str, object]:
    return {
        "label": NOT_BANANA_LABEL,
        "confidence": float(confidence) if confidence is not None else 0.0,
        "top_predictions": [],
        "healthy_probability": None,
        "diseased_probability": None,
        "is_banana_leaf": False,
        "ood_reason": reason,
        "ood_non_plant_score": non_plant_score,
        "mode": "ood",
    }

DISEASE_INFO = {
    "Augmented Banana Black Sigatoka Disease": {
        "status": "Terdeteksi Black Sigatoka (Sigatoka Hitam)",
        "severity": "Tinggi",
        "info": "Penyakit jamur agresif yang menyebabkan bercak hitam pada daun, mempercepat kematian jaringan daun dan mengurangi hasil panen secara drastis.",
        "treatment": "Gunakan fungisida sistemik (Propiconazole/Azoxystrobin), sanitasi daun terinfeksi, perbaiki drainase dan sirkulasi udara.",
    },
    "Augmented Banana Bract Mosaic Virus Disease": {
        "status": "Terdeteksi Bract Mosaic Virus (Virus Mosaik Seludang)",
        "severity": "Tinggi",
        "info": "Penyakit virus yang menyebabkan pola mosaik pada seludang bunga dan daun, ditularkan oleh kutu daun (aphid). Dapat menurunkan kualitas buah.",
        "treatment": "Musnahkan tanaman terinfeksi, kendalikan vektor kutu daun, gunakan bibit bebas virus, dan jaga kebersihan alat pertanian.",
    },
    "Augmented Banana Healthy Leaf": {
        "status": "Daun Terlihat Sehat",
        "severity": "Rendah",
        "info": "Kondisi daun tampak normal tanpa gejala penyakit yang signifikan.",
        "treatment": "Lanjutkan pemantauan rutin dan pemupukan yang seimbang.",
    },
    "Augmented Banana Insect Pest Disease": {
        "status": "Terdeteksi Kerusakan Hama Serangga",
        "severity": "Sedang",
        "info": "Kerusakan daun akibat serangan hama serangga seperti penggulung daun, thrips, atau ulat. Dapat mengurangi luas fotosintesis.",
        "treatment": "Identifikasi jenis hama spesifik, gunakan insektisida yang sesuai atau pengendalian hayati, dan bersihkan gulma di sekitar tanaman.",
    },
    "Augmented Banana Moko Disease": {
        "status": "Terdeteksi Penyakit Moko",
        "severity": "Sangat Tinggi",
        "info": "Penyakit bakteri (Ralstonia solanacearum) yang menyebabkan layu pada tanaman pisang. Sangat menular dan sulit dikendalikan setelah menyebar.",
        "treatment": "Musnahkan tanaman terinfeksi segera, desinfeksi alat tani, hindari penanaman ulang di lahan yang sama selama minimal 12 bulan, dan gunakan bibit bersertifikat.",
    },
    "Augmented Banana Panama Disease": {
        "status": "Terdeteksi Panama Disease (Layu Fusarium)",
        "severity": "Sangat Tinggi",
        "info": "Penyakit tanah yang sangat mematikan (Fusarium oxysporum) bagi tanaman pisang, menyebabkan layu permanen dan tidak ada obat efektif.",
        "treatment": "Isolasi tanaman, jangan memindahkan tanah dari area terinfeksi, gunakan bibit bersertifikat tahan penyakit, dan pertimbangkan rotasi tanaman.",
    },
    "Augmented Banana Yellow Sigatoka Disease": {
        "status": "Terdeteksi Yellow Sigatoka (Sigatoka Kuning)",
        "severity": "Sedang - Tinggi",
        "info": "Penyakit jamur yang menyebabkan garis-garis kuning pada daun dan dapat mematikan jaringan daun. Kurang agresif dibanding Black Sigatoka tetapi tetap merugikan.",
        "treatment": "Sanitasi daun tua, perbaiki drainase, aplikasi fungisida protektif (Mancozeb/Chlorothalonil), dan pantau penyebaran terutama di musim hujan.",
    },
    "Not Banana Leaf": {
        "status": "Bukan Daun Pisang",
        "severity": "-",
        "info": (
            "Gambar yang diunggah sepertinya bukan daun pisang. "
            "Sistem ini hanya dapat menganalisa foto daun pisang."
        ),
        "treatment": (
            "Silakan unggah ulang foto daun pisang yang jelas, dekat, dan dengan "
            "pencahayaan cukup. Hindari foto tangan, wajah, atau objek lain di luar daun pisang."
        ),
    },
}

def load_artifacts() -> tuple[dict[str, object] | None, str | None]:
    """Load model artifacts. Supports ensemble (3 models) and single model fallback."""

    # === Mode Ensemble: 3 model dari ensemble_config.json ===
    if ENSEMBLE_CONFIG_PATH.exists() and LABELS_PATH.exists():
        try:
            config = json.loads(ENSEMBLE_CONFIG_PATH.read_text(encoding="utf-8"))
            all_labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

            models = []
            model_names = []
            for model_info in config["models"]:
                model_path = ARTIFACTS_PATH / model_info["file"]
                if not model_path.exists():
                    return None, f"Model ensemble tidak ditemukan: {model_path}"
                models.append(tf.keras.models.load_model(model_path))
                model_names.append(model_info["name"])

            print(f"[inference] Ensemble loaded: {model_names}")
            return {
                "mode": "ensemble",
                "models": models,
                "model_names": model_names,
                "all_labels": all_labels,
                "config": config,
            }, None
        except Exception as exc:
            return None, f"Gagal load ensemble: {exc}"

    # === Fallback: Single model ===
    if SINGLE_MODEL_PATH.exists() and LABELS_PATH.exists():
        return {
            "mode": "single-stage",
            "model": tf.keras.models.load_model(SINGLE_MODEL_PATH),
            "all_labels": json.loads(LABELS_PATH.read_text(encoding="utf-8")),
        }, None

    return None, (
        f"Model belum ditemukan. Letakkan model ensemble di {ARTIFACTS_PATH} "
        f"(ensemble_config.json + 3 file .keras + labels.json), "
        f"atau single model di {SINGLE_MODEL_PATH}."
    )

def preprocess_image(image: Image.Image) -> np.ndarray:
    rgb_image = image.convert("RGB")
    resized = rgb_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.asarray(resized, dtype=np.float32)
    image_array = image_array / 255.0  # normalisasi ke [0,1] sesuai rescale=1./255 di training
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(artifacts: dict[str, object], image: Image.Image) -> dict[str, object]:
    # === Stage 0: Out-Of-Distribution check ===
    # Hanya blokir jika ImageNet sangat yakin gambar ini adalah objek NON-tumbuhan
    # (tangan, wajah, kendaraan, elektronik, dll).
    # TIDAK blokir hanya karena skor "plant" rendah — daun pisang dari sudut miring
    # atau dengan background tanah sering mendapat skor plant rendah dari ImageNet.
    non_plant_score = _imagenet_non_plant_score(image)
    if non_plant_score is not None and non_plant_score >= OOD_NON_PLANT_BLOCK:
        return _not_banana_payload(
            reason=f"non_plant_score {non_plant_score:.2f} >= {OOD_NON_PLANT_BLOCK}",
            non_plant_score=non_plant_score,
            confidence=None,
        )

    batch = preprocess_image(image)

    # === Mode Ensemble: Soft Voting dari 3 model ===
    if artifacts["mode"] == "ensemble":
        models = artifacts["models"]
        labels = artifacts["all_labels"]

        # Prediksi dari masing-masing model
        predictions = [model.predict(batch, verbose=0)[0] for model in models]

        # Soft Voting: rata-rata probabilitas
        ensemble_pred = np.mean(predictions, axis=0)

        best_index = int(np.argmax(ensemble_pred))
        best_confidence = float(ensemble_pred[best_index])
        top_indices = np.argsort(ensemble_pred)[::-1][:3]

        # Lapis kedua OOD: confidence rendah + non_plant_score tinggi -> blokir
        if (
            non_plant_score is not None
            and best_confidence < OOD_MIN_CONFIDENCE
            and non_plant_score >= OOD_NON_PLANT_BLOCK * 0.5
        ):
            return _not_banana_payload(
                reason=f"low confidence {best_confidence:.2f} + non_plant_score {non_plant_score:.2f}",
                non_plant_score=non_plant_score,
                confidence=best_confidence,
            )

        # Per-model detail (opsional, untuk debugging/UI)
        per_model = {}
        model_names = artifacts.get("model_names", [f"model_{i}" for i in range(len(models))])
        for name, pred in zip(model_names, predictions):
            idx = int(np.argmax(pred))
            per_model[name] = {
                "label": labels[idx],
                "confidence": float(pred[idx]),
            }

        return {
            "label": labels[best_index],
            "confidence": best_confidence,
            "top_predictions": [(labels[index], float(ensemble_pred[index])) for index in top_indices],
            "healthy_probability": float(ensemble_pred[labels.index("Augmented Banana Healthy Leaf")]) if "Augmented Banana Healthy Leaf" in labels else None,
            "diseased_probability": None,
            "is_banana_leaf": True,
            "ood_non_plant_score": non_plant_score,
            "per_model": per_model,
            "mode": "ensemble",
        }

    # === Fallback: Single model ===
    model = artifacts["model"]
    labels = artifacts["all_labels"]
    predictions = model.predict(batch, verbose=0)[0]
    best_index = int(np.argmax(predictions))
    top_indices = np.argsort(predictions)[::-1][:3]
    best_confidence = float(predictions[best_index])
    if (
        non_plant_score is not None
        and best_confidence < OOD_MIN_CONFIDENCE
        and non_plant_score >= OOD_NON_PLANT_BLOCK * 0.5
    ):
        return _not_banana_payload(
            reason=f"low confidence {best_confidence:.2f} + non_plant_score {non_plant_score:.2f}",
            non_plant_score=non_plant_score,
            confidence=best_confidence,
        )
    return {
        "label": labels[best_index],
        "confidence": best_confidence,
        "top_predictions": [(labels[index], float(predictions[index])) for index in top_indices],
        "healthy_probability": None,
        "diseased_probability": None,
        "is_banana_leaf": True,
        "ood_non_plant_score": non_plant_score,
        "mode": "single-stage",
    }
