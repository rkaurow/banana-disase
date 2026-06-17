import os
import glob
import sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.inference import load_artifacts, predict_image

artifacts, _ = load_artifacts()

fixture_dir = "tests/banana_leaf"
healthy_dir = "datasets/Augmented Banana Healthy Leaf"

patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
files = []
for directory in [fixture_dir, healthy_dir]:
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern)))
    if files:
        break

if artifacts is None:
    raise SystemExit("Artifacts belum tersedia.")

if not files:
    raise SystemExit(
        "Tidak ada fixture daun pisang. Tambahkan minimal 5 gambar nyata ke tests/banana_leaf/ "
        "atau siapkan datasets/Augmented Banana Healthy Leaf."
    )

print(f"Testing {min(5, len(files))} healthy images...")
for f in files[:5]:
    img = Image.open(f)
    pred = predict_image(artifacts, img)
    print(f"File: {os.path.basename(f)} -> Pred: {pred['label']} ({pred['confidence']:.2f})")
    if pred.get("is_banana_leaf") is False:
        raise SystemExit(f"GAGAL: {f} mental ke Not Banana Leaf")
