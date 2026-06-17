"""Uji empiris: apa yang terjadi jika sistem diberi foto NON-pisang.

Menjalankan pipeline inferensi lewat predict_image. Jika artifact two-stage sudah
tersedia, keputusan utama berasal dari banana gate; jika belum, test memakai
fallback ensemble legacy.
"""
import sys
import os
from pathlib import Path

from PIL import Image

# pastikan src/ bisa di-import saat dijalankan dari root repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import (
    load_artifacts,
    predict_image,
    OOD_NON_PLANT_BLOCK,
    OOD_DISAGREE_NON_PLANT,
)


def run(image_path: str) -> None:
    print(f"\n=== Uji: {image_path} ===")
    image = Image.open(image_path)

    if os.getenv("CHECK_IMAGENET_OOD") == "1":
        from src.inference import _imagenet_non_plant_score

        non_plant = _imagenet_non_plant_score(image)
        print(f"non_plant_score (ImageNet)   : {non_plant:.4f}" if non_plant is not None else "non_plant_score: N/A")
        print(f"  ambang blokir lapis-1       : {OOD_NON_PLANT_BLOCK}")
        print(f"  ambang disagreement lapis-3 : {OOD_DISAGREE_NON_PLANT}")

    artifacts, err = load_artifacts()
    if err:
        print(f"ERROR load_artifacts: {err}")
        return

    result = predict_image(artifacts, image)
    print(f"\nLABEL AKHIR  : {result['label']}")
    print(f"confidence   : {result['confidence']:.4f}")
    print(f"is_banana_leaf: {result['is_banana_leaf']}")
    if result.get("banana_probability") is not None:
        print(f"banana_prob  : {result['banana_probability']:.4f}")
    if result.get("banana_threshold") is not None:
        print(f"gate_threshold: {result['banana_threshold']:.4f}")
    if result.get("ood_reason"):
        print(f"ood_reason   : {result['ood_reason']}")
    if result.get("per_model"):
        print("per-model (sinyal kesepakatan):")
        for name, v in result["per_model"].items():
            print(f"  - {name:12s}: {v['label']}  ({v['confidence']:.3f})")
    if result.get("top_predictions"):
        print("top-3 ensemble:")
        for lbl, p in result["top_predictions"]:
            print(f"  - {lbl}  ({p:.3f})")


if __name__ == "__main__":
    default_paths = ["tests/daun_pepaya.jpeg", "tests/daun_kelapa.jpeg"]
    paths = sys.argv[1:] or [p for p in default_paths if Path(p).exists()]
    if not paths:
        raise SystemExit("Tidak ada file uji. Siapkan tests/daun_pepaya.jpeg atau tests/daun_kelapa.jpeg.")
    for p in paths:
        run(p)
