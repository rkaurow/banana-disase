import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# Konfigurasi
DATASET_PATH = Path("datasets")
ARTIFACTS_PATH = Path("artifacts")
MODEL_PATH = ARTIFACTS_PATH / "banana_disease_model.keras"
LABELS_PATH = ARTIFACTS_PATH / "labels.json"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def evaluate_model():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        print("Error: Model atau labels tidak ditemukan. Jalankan train.py dulu.")
        return

    # Load labels
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")

    # Data Generator untuk Evaluasi (tanpa augmentasi, hanya rescale)
    test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    test_generator = test_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False # Penting untuk confusion matrix
    )

    # Prediksi
    print("Mengevaluasi model pada data validasi...")
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    
    # Classification Report
    print("\nClassification Report:")
    report = classification_report(test_generator.classes, y_pred, target_names=labels)
    print(report)
    
    with open(ARTIFACTS_PATH / "evaluation_report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(test_generator.classes, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Banana Disease Detection')
    plt.savefig(ARTIFACTS_PATH / "confusion_matrix.png")
    print(f"Confusion matrix disimpan di {ARTIFACTS_PATH / 'confusion_matrix.png'}")
    plt.close()

if __name__ == "__main__":
    evaluate_model()
