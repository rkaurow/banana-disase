from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("artifacts/banana_disease_model.keras")
LABELS_PATH = Path("artifacts/labels.json")
BINARY_MODEL_PATH = Path("artifacts/binary_model.keras")
BINARY_LABELS_PATH = Path("artifacts/binary_labels.json")
DISEASE_MODEL_PATH = Path("artifacts/disease_model.keras")
DISEASE_LABELS_PATH = Path("artifacts/disease_labels.json")
CONFIG_PATH = Path("artifacts/training_config.json")
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
    "Panama Disease": {
        "status": "Terdeteksi Panama Disease (Layu Fusarium)",
        "severity": "Tinggi",
        "info": "Penyakit tanah yang sangat mematikan bagi tanaman pisang, menyebabkan layu permanen.",
        "treatment": "Isolasi tanaman, jangan memindahkan tanah dari area terinfeksi, dan gunakan bibit bersertifikat.",
    },
    "Cordana": {
        "status": "Terdeteksi Cordana Leaf Spot",
        "severity": "Sedang",
        "info": "Bercak daun berbentuk oval dengan pusat abu-abu yang dapat mengurangi luas fotosintesis.",
        "treatment": "Kurangi kelembapan, perbaiki sirkulasi udara, dan buang daun yang terinfeksi berat.",
    },
    "Yellow and Black Sigatoka": {
        "status": "Terdeteksi Sigatoka (Kuning/Hitam)",
        "severity": "Sedang - Tinggi",
        "info": "Penyakit jamur yang menyebabkan garis-garis pada daun dan dapat mematikan jaringan daun dengan cepat.",
        "treatment": "Sanitasi daun tua, perbaiki drainase, dan pantau penyebaran terutama di musim hujan.",
    },
    "Healthy": {
        "status": "Daun Terlihat Sehat",
        "severity": "Rendah",
        "info": "Kondisi daun tampak normal tanpa gejala penyakit yang signifikan.",
        "treatment": "Lanjutkan pemantauan rutin dan pemupukan yang seimbang.",
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
    has_two_stage = all(
        path.exists()
        for path in [BINARY_MODEL_PATH, BINARY_LABELS_PATH, DISEASE_MODEL_PATH, DISEASE_LABELS_PATH]
    )
    if has_two_stage:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
        all_labels = json.loads(LABELS_PATH.read_text(encoding="utf-8")) if LABELS_PATH.exists() else []
        return {
            "mode": "two-stage",
            "binary_model": tf.keras.models.load_model(BINARY_MODEL_PATH),
            "binary_labels": json.loads(BINARY_LABELS_PATH.read_text(encoding="utf-8")),
            "disease_model": tf.keras.models.load_model(DISEASE_MODEL_PATH),
            "disease_labels": json.loads(DISEASE_LABELS_PATH.read_text(encoding="utf-8")),
            "healthy_class": config.get("healthy_class", "Healthy"),
            "diseased_label": config.get("diseased_label", "Diseased"),
            "all_labels": all_labels,
        }, None

    if not MODEL_PATH.exists():
        return None, f"Model belum ditemukan di {MODEL_PATH}. Latih model pisang terlebih dahulu."
    if not LABELS_PATH.exists():
        return None, f"File label belum ditemukan di {LABELS_PATH}."

    return {
        "mode": "single-stage",
        "model": tf.keras.models.load_model(MODEL_PATH),
        "all_labels": json.loads(LABELS_PATH.read_text(encoding="utf-8")),
    }, None

def preprocess_image(image: Image.Image) -> np.ndarray:
    rgb_image = image.convert("RGB")
    resized = rgb_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.asarray(resized, dtype=np.float32)
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
    if artifacts["mode"] == "two-stage":
        binary_model = artifacts["binary_model"]
        disease_model = artifacts["disease_model"]
        binary_labels = artifacts["binary_labels"]
        disease_labels = artifacts["disease_labels"]
        healthy_class = artifacts["healthy_class"]
        diseased_label = artifacts["diseased_label"]

        binary_predictions = binary_model.predict(batch, verbose=0)[0]
        disease_predictions = disease_model.predict(batch, verbose=0)[0]
        binary_lookup = {label: index for index, label in enumerate(binary_labels)}
        healthy_probability = float(binary_predictions[binary_lookup[healthy_class]])
        diseased_probability = float(binary_predictions[binary_lookup[diseased_label]])
        disease_index = int(np.argmax(disease_predictions))
        disease_probability = float(disease_predictions[disease_index])
        predicted_label = healthy_class if healthy_probability >= diseased_probability else disease_labels[disease_index]
        top_indices = np.argsort(disease_predictions)[::-1][:3]
        final_confidence = (
            healthy_probability
            if predicted_label == healthy_class
            else diseased_probability * disease_probability
        )
        # Lapis kedua: model pisang sangat tidak yakin DAN non_plant_score cukup tinggi -> blokir.
        if (
            non_plant_score is not None
            and final_confidence < OOD_MIN_CONFIDENCE
            and non_plant_score >= OOD_NON_PLANT_BLOCK * 0.5
        ):
            return _not_banana_payload(
                reason=f"low confidence {final_confidence:.2f} + non_plant_score {non_plant_score:.2f}",
                non_plant_score=non_plant_score,
                confidence=final_confidence,
            )
        return {
            "label": predicted_label,
            "confidence": final_confidence,
            "top_predictions": [(disease_labels[index], float(disease_predictions[index])) for index in top_indices],
            "healthy_probability": healthy_probability,
            "diseased_probability": diseased_probability,
            "is_banana_leaf": True,
            "ood_non_plant_score": non_plant_score,
            "mode": "two-stage",
        }

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
