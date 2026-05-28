import os
import json
import ssl
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

# Fix SSL
ssl._create_default_https_context = ssl._create_unverified_context

DATASET_PATH = Path("datasets")
ARTIFACTS_PATH = Path("artifacts")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def plot_history(history, filename):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(len(acc))
    plt.figure(figsize=(8, 4))
    plt.plot(epochs_range, acc, label='Train Acc')
    plt.plot(epochs_range, val_acc, label='Val Acc')
    plt.legend()
    plt.title(f'History - {filename}')
    plt.savefig(ARTIFACTS_PATH / filename)
    plt.close()

def build_model(num_classes, label_smoothing=0.0):
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model, base_model


def _compute_class_weight(generator):
    y = generator.classes
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
    cw = {i: float(w) for i, w in enumerate(weights)}
    labels = list(generator.class_indices.keys())
    print(f"Class weights: { {labels[i]: round(w, 3) for i, w in cw.items()} }")
    return cw

def train_binary():
    print("\n--- Training Stage 1: Binary (Healthy vs Diseased) ---")
    
    # Kita buat temporary directory untuk memisahkan Healthy dan Sakit
    temp_dir = Path("temp_binary_data")
    temp_dir.mkdir(exist_ok=True)
    (temp_dir / "Healthy").mkdir(exist_ok=True)
    (temp_dir / "Diseased").mkdir(exist_ok=True)

    # Link atau copy data (link lebih hemat disk jika didukung OS)
    for class_folder in DATASET_PATH.iterdir():
        if not class_folder.is_dir(): continue
        target = temp_dir / "Healthy" if class_folder.name == "Healthy" else temp_dir / "Diseased"
        for img in class_folder.glob("*"):
            # Pakai symbolic link jika memungkinkan, jika tidak copy
            try:
                os.symlink(img.absolute(), target / img.name)
            except:
                shutil.copy(img, target / img.name)

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True, rotation_range=20)
    train_gen = datagen.flow_from_directory(temp_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, subset='training')
    val_gen = datagen.flow_from_directory(temp_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, subset='validation')

    # Healthy (~1001) jauh lebih sedikit dari Diseased (~3674) -> wajib pakai class_weight
    class_weight = _compute_class_weight(train_gen)

    model, _ = build_model(2, label_smoothing=0.05)
    history = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        class_weight=class_weight,
        verbose=1,
    )
    
    model.save(ARTIFACTS_PATH / "binary_model.keras")
    with open(ARTIFACTS_PATH / "binary_labels.json", "w") as f:
        json.dump(list(train_gen.class_indices.keys()), f)
    
    plot_history(history, "binary_history.png")
    shutil.rmtree(temp_dir) # Hapus temp data

def train_diseased_only():
    print("\n--- Training Stage 2: Disease Classification (Cordana, Panama, Sigatoka) ---")
    
    # Buat temp dir hanya untuk kelas yang sakit
    temp_dir = Path("temp_disease_data")
    temp_dir.mkdir(exist_ok=True)
    for class_folder in DATASET_PATH.iterdir():
        if class_folder.is_dir() and class_folder.name != "Healthy":
            shutil.copytree(class_folder, temp_dir / class_folder.name, dirs_exist_ok=True)

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, horizontal_flip=True, rotation_range=30)
    train_gen = datagen.flow_from_directory(temp_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, subset='training')
    val_gen = datagen.flow_from_directory(temp_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, subset='validation')

    labels = list(train_gen.class_indices.keys())
    class_weight = _compute_class_weight(train_gen)
    model, _ = build_model(len(labels), label_smoothing=0.05)
    history = model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        class_weight=class_weight,
        verbose=1,
    )
    
    model.save(ARTIFACTS_PATH / "disease_model.keras")
    with open(ARTIFACTS_PATH / "disease_labels.json", "w") as f:
        json.dump(labels, f)
    
    plot_history(history, "disease_only_history.png")
    shutil.rmtree(temp_dir)

def create_config():
    config = {
        "healthy_class": "Healthy",
        "diseased_label": "Diseased"
    }
    with open(ARTIFACTS_PATH / "training_config.json", "w") as f:
        json.dump(config, f)

if __name__ == "__main__":
    ARTIFACTS_PATH.mkdir(exist_ok=True)
    create_config()
    train_binary()
    train_diseased_only()
    print("\nSemua model Two-Stage berhasil dilatih dan disimpan!")
