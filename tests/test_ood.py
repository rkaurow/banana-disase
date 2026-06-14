import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# Buat gambar dummy hitam dengan teks putih (mirip IDE dark mode)
img = Image.new('RGB', (224, 224), color = (30, 30, 30))
d = ImageDraw.Draw(img)
d.text((10,10), "def function_name():", fill=(200,200,200))
d.text((10,30), "    print('Hello World')", fill=(200,200,200))
d.text((10,50), "    return 0", fill=(200,200,200))

model = tf.keras.applications.MobileNetV2(weights="imagenet")
arr = np.asarray(img, dtype=np.float32)
arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
arr = np.expand_dims(arr, axis=0)

preds = model.predict(arr, verbose=0)
decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=20)[0]

print("Top 20 classes for dark IDE screenshot:")
for i, (id, name, prob) in enumerate(decoded):
    print(f"{i+1}. {name} ({prob:.3f})")
