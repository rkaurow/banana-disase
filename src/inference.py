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
        return {
            "label": predicted_label,
            "confidence": healthy_probability if predicted_label == healthy_class else diseased_probability * disease_probability,
            "top_predictions": [(disease_labels[index], float(disease_predictions[index])) for index in top_indices],
            "healthy_probability": healthy_probability,
            "diseased_probability": diseased_probability,
            "mode": "two-stage",
        }

    model = artifacts["model"]
    labels = artifacts["all_labels"]
    predictions = model.predict(batch, verbose=0)[0]
    best_index = int(np.argmax(predictions))
    top_indices = np.argsort(predictions)[::-1][:3]
    return {
        "label": labels[best_index],
        "confidence": float(predictions[best_index]),
        "top_predictions": [(labels[index], float(predictions[index])) for index in top_indices],
        "healthy_probability": None,
        "diseased_probability": None,
        "mode": "single-stage",
    }
