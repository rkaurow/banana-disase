import os
import glob
from PIL import Image
from src.inference import load_artifacts, predict_image

artifacts, _ = load_artifacts()

healthy_dir = "datasets/Augmented Banana Healthy Leaf"
files = glob.glob(os.path.join(healthy_dir, "*.jpg")) + glob.glob(os.path.join(healthy_dir, "*.png"))

print(f"Testing {min(5, len(files))} healthy images...")
for f in files[:5]:
    img = Image.open(f)
    pred = predict_image(artifacts, img)
    print(f"File: {os.path.basename(f)} -> Pred: {pred['label']} ({pred['confidence']:.2f})")
