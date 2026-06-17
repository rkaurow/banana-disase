import io
import os
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

from src.inference import load_artifacts, predict_image, DISEASE_INFO
from src.ai_response import generate_disease_response, chat_with_bot

root_path = os.getenv("ROOT_PATH", "")
app = FastAPI(title="Banana Disease API", root_path=root_path)

ARTIFACT_FILES = [
    Path("artifacts/ensemble_config.json"),
    Path("artifacts/labels.json"),
    Path("artifacts/banana_gate.keras"),
    Path("artifacts/banana_gate_config.json"),
    Path("artifacts/model_cnn.keras"),
    Path("artifacts/model_resnet.keras"),
    Path("artifacts/model_inception.keras"),
    Path("artifacts/banana_disease_model.keras"),
]

_artifacts_lock = Lock()
_artifacts_signature = None
artifacts = None
error_msg = None


def _artifact_signature():
    signature = []
    for path in ARTIFACT_FILES:
        if path.exists():
            stat = path.stat()
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        else:
            signature.append((str(path), None, None))
    return tuple(signature)


def get_current_artifacts():
    """Load artifacts at startup and reload them if model/config files changed."""
    global artifacts, error_msg, _artifacts_signature

    signature = _artifact_signature()
    with _artifacts_lock:
        if artifacts is None or signature != _artifacts_signature:
            artifacts, error_msg = load_artifacts()
            _artifacts_signature = signature
            if error_msg:
                print(f"WARNING: {error_msg}")
    return artifacts, error_msg


get_current_artifacts()

class ChatRequest(BaseModel):
    messages: list[dict[str, str]]

import asyncio

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    current_artifacts, current_error = get_current_artifacts()
    if current_artifacts is None:
        raise HTTPException(status_code=503, detail="Model is not loaded properly.")
    
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    def process_prediction():
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise ValueError(f"Invalid image file: {e}")

        prediction = predict_image(current_artifacts, image)
        
        # Add Disease Info logic
        label = prediction["label"]
        disease = DISEASE_INFO.get(label, {
            "status": label,
            "severity": "Perlu ditinjau",
            "info": "Informasi detail belum tersedia.",
            "treatment": "Lakukan pengecekan lapangan."
        })
        prediction["disease_info"] = disease

        # Generate AI response (skip kalau bukan daun pisang -> hemat token & cegah jawaban ngaco)
        if prediction.get("is_banana_leaf") is False:
            prediction["ai_response"] = {
                "headline": "Bukan Daun Pisang",
                "summary": (
                    "Sistem mendeteksi bahwa gambar yang Anda unggah kemungkinan besar "
                    "bukan daun pisang. Mohon unggah ulang foto daun pisang yang jelas."
                ),
                "meaning": (
                    "Model deteksi penyakit ini hanya dilatih untuk daun pisang. "
                    "Gambar lain (tangan, wajah, benda, daun tanaman lain) tidak dapat dianalisa."
                ),
                "actions": [
                    "Ambil foto daun pisang dari jarak dekat (30-60 cm).",
                    "Pastikan seluruh helai daun terlihat dan fokus.",
                    "Gunakan pencahayaan alami yang cukup, hindari bayangan keras.",
                    "Hindari memotret tangan, wajah, atau objek selain daun pisang.",
                    "Coba beberapa sudut: atas daun, bawah daun, dan area yang dicurigai sakit.",
                ],
                "prevention": [
                    "Selalu verifikasi objek pada foto sebelum unggah.",
                    "Bersihkan lensa kamera agar gambar tidak buram.",
                    "Gunakan satu daun per foto untuk hasil terbaik.",
                ],
                "warning": "Hasil ini bukan diagnosis penyakit — silakan unggah ulang foto daun pisang.",
            }
        else:
            try:
                ai_response = generate_disease_response(prediction)
                prediction["ai_response"] = ai_response
            except Exception as e:
                print(f"Error generating AI response: {e}")
                prediction["ai_response"] = None

        return prediction

    try:
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(None, process_prediction)
        return JSONResponse(content=prediction)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-status")
async def model_status():
    current_artifacts, current_error = get_current_artifacts()
    if current_artifacts is None:
        return {
            "loaded": False,
            "error": current_error,
            "cwd": str(Path.cwd()),
        }
    return {
        "loaded": True,
        "mode": current_artifacts.get("mode"),
        "labels_count": len(current_artifacts.get("all_labels", [])),
        "model_names": current_artifacts.get("model_names"),
        "banana_threshold": (
            current_artifacts.get("banana_gate_config", {}).get("banana_threshold")
            if current_artifacts.get("banana_gate_config")
            else None
        ),
        "cwd": str(Path.cwd()),
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    response = chat_with_bot(request.messages)
    if response:
        return {"response": response}
    raise HTTPException(status_code=500, detail="Failed to get AI response")

if not os.path.exists("frontend"):
    os.makedirs("frontend")

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
