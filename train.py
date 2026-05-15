import os
import json
import ssl
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# Fix SSL Certificate Error on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Konfigurasi
DATASET_PATH = Path("datasets")
ARTIFACTS_PATH = Path("artifacts")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

def plot_history(history, filename):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(ARTIFACTS_PATH / filename)
    plt.close()

def train_model():
    if not DATASET_PATH.exists():
        print(f"Error: Folder dataset '{DATASET_PATH}' tidak ditemukan.")
        return

    # Data Augmentation yang lebih kuat untuk mengurangi overfitting
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Simpan labels
    labels = list(train_generator.class_indices.keys())
    with open(ARTIFACTS_PATH / "labels.json", "w") as f:
        json.dump(labels, f)
    print(f"Labels disimpan: {labels}")

    # Base Model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3), # Ditambah untuk mengurangi overfitting
        layers.Dense(len(labels), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        ARTIFACTS_PATH / "best_model.keras",
        save_best_only=True,
        monitor='val_accuracy'
    )
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )

    print("--- Phase 1: Training Head ---")
    history = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    plot_history(history, "history_phase1.png")

    print("--- Phase 2: Fine-Tuning ---")
    # Unfreeze top layers of base model
    base_model.trainable = True
    # Fine-tune dari layer ke-100 ke atas
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # LR sangat kecil untuk fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_fine = model.fit(
        train_generator,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    plot_history(history_fine, "history_phase2.png")

    # Simpan model akhir
    model.save(ARTIFACTS_PATH / "banana_disease_model.keras")
    print(f"Model berhasil disimpan di artifacts/banana_disease_model.keras")

if __name__ == "__main__":
    ARTIFACTS_PATH.mkdir(exist_ok=True)
    train_model()
