from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

SUMOPOD_API_KEY_ENV = "SUMOPOD_API_KEY"
DEFAULT_BASE_URL = "https://ai.sumopod.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HEALTHY_LABEL = "Healthy"

LABEL_PROMPT_HINTS = {
    "Panama Disease": "Fokus pada penyakit layu Fusarium (Panama), risiko penyebaran tanah, pentingnya isolasi, dan pembersihan alat tani.",
    "Cordana": "Fokus pada bercak daun Cordana, pengaruh kelembapan, dan pentingnya sirkulasi udara di area pisang.",
    "Yellow Sigatoka": "Fokus pada penyakit bercak daun kuning (Sigatoka), dampak pada hasil buah, dan sanitasi daun tua.",
    "Black Sigatoka": "Fokus pada Sigatoka Hitam yang lebih agresif, pengurangan kelembapan, dan monitoring penyebaran cepat.",
    "Healthy": "Fokus pada pemeliharaan rutin, monitoring berkala, dan menjaga kebersihan area kebun.",
}


FINAL_SYSTEM_PROMPT = """Kamu adalah asisten agrikultur ahli penyakit pisang untuk aplikasi deteksi daun pisang.

Tugasmu adalah menjelaskan hasil prediksi model computer vision dalam Bahasa Indonesia yang profesional, praktis, dan SANGAT SPESIFIK.

Aturan Utama:
- WAJIB memberikan rekomendasi produk spesifik (Merek Fungisida, Bakterisida, atau Desinfektan) yang umum beredar di Indonesia sesuai dengan penyakit yang terdeteksi.
- Sebutkan merek populer seperti Amistar Top, Dithane M-45, Nordox, Copcide, atau Skor jika relevan dengan penyakitnya.
- Berikan saran penanganan (actions) minimal 5 poin yang mendetail, termasuk cara aplikasi produk tersebut (misal: penyemprotan foliar atau pengocoran tanah).
- Jika penyakitnya adalah Panama Disease (Layu Fusarium), tekankan penggunaan agens hayati seperti Trichoderma atau desinfektan alat pertanian.
- Jika penyakitnya adalah Sigatoka atau Cordana, fokus pada fungisida sistemik dan kontak.
- Tetap berikan peringatan untuk membaca label dosis pada kemasan produk.
- Kembalikan JSON valid tanpa markdown, tanpa penjelasan tambahan di luar JSON.

Format JSON yang wajib:
{
  "headline": "string",
  "summary": "string",
  "meaning": "string",
  "actions": ["Contoh: Semprotkan Fungisida [Merek] dengan bahan aktif [A]...", "string", "string", "string", "string"],
  "prevention": ["string", "string", "string"],
  "warning": "string"
}
"""


def get_ai_runtime_status() -> dict[str, Any]:
    api_key, api_key_source = _get_setting(SUMOPOD_API_KEY_ENV)
    base_url, _ = _get_setting("SUMOPOD_BASE_URL", DEFAULT_BASE_URL)
    model, _ = _get_setting("SUMOPOD_MODEL", DEFAULT_MODEL)
    try:
        import openai  # noqa: F401
        package_available = True
    except ImportError:
        package_available = False
    return {
        "enabled": bool(api_key) and package_available,
        "api_key_present": bool(api_key),
        "package_available": package_available,
        "api_key_env": SUMOPOD_API_KEY_ENV,
        "api_key_source": api_key_source,
        "base_url": base_url,
        "model": model,
    }


def _read_dotenv() -> dict[str, str]:
    dotenv_path = PROJECT_ROOT / ".env"
    if not dotenv_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _read_streamlit_secrets() -> dict[str, str]:
    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return {}

    parsed = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
    return {key: str(value).strip() for key, value in parsed.items() if isinstance(value, (str, int, float, bool))}


def _get_setting(name: str, default: str = "") -> tuple[str, str]:
    env_value = os.getenv(name, "").strip()
    if env_value:
        return env_value, "environment"

    dotenv_value = _read_dotenv().get(name, "").strip()
    if dotenv_value:
        return dotenv_value, ".env"

    secret_value = _read_streamlit_secrets().get(name, "").strip()
    if secret_value:
        return secret_value, ".streamlit/secrets.toml"

    return default, "default" if default else "missing"


def _format_top_predictions(top_predictions: list[tuple[str, float]]) -> str:
    lines = []
    for index, (label, score) in enumerate(top_predictions, start=1):
        lines.append(f"{index}. {label} - {score * 100:.2f}%")
    return "\n".join(lines)


def should_generate_ai_response(prediction: dict[str, Any]) -> bool:
    label = str(prediction.get("label", "")).strip()
    if not label or label == HEALTHY_LABEL:
        return False
    return True


def _build_label_prompt_hint(label: str) -> str:
    return LABEL_PROMPT_HINTS.get(
        label,
        "Fokus pada arti hasil model, langkah awal yang aman, pencegahan dasar, dan kebutuhan verifikasi lapangan.",
    )


def build_user_prompt(prediction: dict[str, Any]) -> str:
    label = str(prediction["label"])
    confidence = float(prediction["confidence"]) * 100
    if confidence >= 90:
        confidence_note = "tinggi"
    elif confidence >= 75:
        confidence_note = "menengah"
    else:
        confidence_note = "rendah"

    lines = [
        "Berikut hasil prediksi model deteksi penyakit daun pisang:",
        "",
        f"Label utama: {label}",
        f"Confidence utama: {confidence:.2f}%",
        f"Tingkat keyakinan: {confidence_note}",
        "Top 3 prediksi:",
        _format_top_predictions(prediction["top_predictions"]),
        "",
        f"Mode model: {prediction['mode']}",
    ]

    healthy_probability = prediction.get("healthy_probability")
    diseased_probability = prediction.get("diseased_probability")
    if healthy_probability is not None and diseased_probability is not None:
        lines.extend(
            [
                f"Confidence healthy: {float(healthy_probability) * 100:.2f}%",
                f"Confidence diseased: {float(diseased_probability) * 100:.2f}%",
            ]
        )

    lines.extend(
        [
            "",
            "Tolong buat penjelasan hasil ini dalam Bahasa Indonesia untuk user aplikasi pertanian.",
            _build_label_prompt_hint(label),
            "Jika confidence utama di bawah 85%, warning harus menekankan bahwa hasil belum pasti dan perlu verifikasi ulang.",
            "Batasi jawaban agar tetap ringkas dan praktis.",
        ]
    )
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def _normalize_response(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "headline": str(data.get("headline", "Hasil analisa AI")).strip(),
        "summary": str(data.get("summary", "")).strip(),
        "meaning": str(data.get("meaning", "")).strip(),
        "actions": [str(item).strip() for item in data.get("actions", []) if str(item).strip()],
        "prevention": [str(item).strip() for item in data.get("prevention", []) if str(item).strip()],
        "warning": str(data.get("warning", "")).strip(),
    }


def get_sumopod_client() -> Any | None:
    runtime = get_ai_runtime_status()
    if not runtime["enabled"]:
        return None

    from openai import OpenAI

    return OpenAI(
        api_key=_get_setting(SUMOPOD_API_KEY_ENV)[0],
        base_url=str(runtime["base_url"]),
        timeout=15.0,
    )


def generate_disease_response(prediction: dict[str, Any]) -> dict[str, Any] | None:
    if not should_generate_ai_response(prediction):
        return None

    runtime = get_ai_runtime_status()
    client = get_sumopod_client()
    if client is None:
        return None

    response = client.chat.completions.create(
        model=str(runtime["model"]),
        messages=[
            {"role": "system", "content": FINAL_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(prediction)},
        ],
        max_tokens=800,
        temperature=0.4,
    )

    content = response.choices[0].message.content or ""
    parsed = _extract_json(content)
    if parsed is None:
        return {
            "headline": "Penjelasan AI",
            "summary": content.strip(),
            "meaning": "",
            "actions": [],
            "prevention": [],
            "warning": "Respons AI tidak dalam format JSON yang diharapkan.",
        }
    return _normalize_response(parsed)


def generate_ai_response(prediction: dict[str, Any]) -> dict[str, Any] | None:
    return generate_disease_response(prediction)


def chat_with_bot(messages: list[dict[str, str]]) -> str | None:
    runtime = get_ai_runtime_status()
    client = get_sumopod_client()
    if client is None:
        return None

    try:
        response = client.chat.completions.create(
            model=str(runtime["model"]),
            messages=messages,
            max_tokens=800,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

