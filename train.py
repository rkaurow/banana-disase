import os
import json
import ssl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score

# Fix SSL Certificate Error on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Konfigurasi
DATASET_PATH = Path("datasets")
ARTIFACTS_PATH = Path("artifacts")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LABEL_SMOOTHING = 0.05
NUM_CLASSES = 7


def get_callbacks(model_name):
    """Callbacks untuk setiap model: EarlyStopping, Checkpoint, ReduceLR."""
    return [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=str(ARTIFACTS_PATH / f"best_{model_name}.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]


def plot_history(history, model_name):
    """Plot training history (accuracy & loss)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title(f'{model_name} — Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title(f'{model_name} — Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    plt.savefig(ARTIFACTS_PATH / f"history_{safe_name}.png", dpi=150)
    plt.close()


def build_cnn_model(num_classes):
    """Custom CNN dari notebook (8 Conv layers + BatchNorm + Dropout)."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def build_resnet_model(num_classes):
    """ResNet50 transfer learning (frozen base)."""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def build_inception_model(num_classes):
    """InceptionV3 transfer learning (frozen base)."""
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_model():
    if not DATASET_PATH.exists():
        print(f"Error: Folder dataset '{DATASET_PATH}' tidak ditemukan.")
        return

    # Data Augmentation
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

    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Simpan labels
    labels = list(train_generator.class_indices.keys())
    num_classes = len(labels)
    with open(ARTIFACTS_PATH / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Labels disimpan ({num_classes}): {labels}")

    # Class weights (dataset tidak seimbang)
    y_train = train_generator.classes
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight = {i: float(w) for i, w in enumerate(class_weights_array)}
    print(f"Class weights: { {labels[i]: round(w, 3) for i, w in class_weight.items()} }")

    # Loss function with label smoothing
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

    # ===== Model 1: Custom CNN =====
    print(f"\n{'='*60}")
    print("Training Model 1: Custom CNN")
    print(f"{'='*60}")

    model_cnn = build_cnn_model(num_classes)
    model_cnn.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    history_cnn = model_cnn.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        class_weight=class_weight,
        callbacks=get_callbacks('cnn'),
        verbose=1
    )
    plot_history(history_cnn, 'Custom CNN')
    model_cnn.save(ARTIFACTS_PATH / "model_cnn.keras")

    # ===== Model 2: ResNet50 =====
    print(f"\n{'='*60}")
    print("Training Model 2: ResNet50")
    print(f"{'='*60}")

    model_resnet = build_resnet_model(num_classes)
    model_resnet.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    history_resnet = model_resnet.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        class_weight=class_weight,
        callbacks=get_callbacks('resnet'),
        verbose=1
    )
    plot_history(history_resnet, 'ResNet50')
    model_resnet.save(ARTIFACTS_PATH / "model_resnet.keras")

    # ===== Model 3: InceptionV3 =====
    print(f"\n{'='*60}")
    print("Training Model 3: InceptionV3")
    print(f"{'='*60}")

    model_inception = build_inception_model(num_classes)
    model_inception.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

    history_inception = model_inception.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        class_weight=class_weight,
        callbacks=get_callbacks('inception'),
        verbose=1
    )
    plot_history(history_inception, 'InceptionV3')
    model_inception.save(ARTIFACTS_PATH / "model_inception.keras")

    # ===== Ensemble Evaluation =====
    print(f"\n{'='*60}")
    print("Evaluasi Ensemble (Soft Voting)")
    print(f"{'='*60}")

    y_true = val_generator.classes
    prob_cnn = model_cnn.predict(val_generator, verbose=0)
    prob_resnet = model_resnet.predict(val_generator, verbose=0)
    prob_inception = model_inception.predict(val_generator, verbose=0)

    ensemble_prob = (prob_cnn + prob_resnet + prob_inception) / 3
    y_pred_ensemble = np.argmax(ensemble_prob, axis=1)

    acc_cnn = accuracy_score(y_true, np.argmax(prob_cnn, axis=1))
    acc_resnet = accuracy_score(y_true, np.argmax(prob_resnet, axis=1))
    acc_inception = accuracy_score(y_true, np.argmax(prob_inception, axis=1))
    acc_ensemble = accuracy_score(y_true, y_pred_ensemble)

    print(f"\n  Custom CNN:             {acc_cnn*100:.2f}%")
    print(f"  ResNet50:               {acc_resnet*100:.2f}%")
    print(f"  InceptionV3:            {acc_inception*100:.2f}%")
    print(f"  Ensemble (Soft Voting): {acc_ensemble*100:.2f}%")

    report = classification_report(y_true, y_pred_ensemble, target_names=labels)
    print(f"\nClassification Report (Ensemble):\n{report}")

    # Simpan ensemble config
    ensemble_config = {
        "models": [
            {"name": "Custom CNN", "file": "model_cnn.keras"},
            {"name": "ResNet50", "file": "model_resnet.keras"},
            {"name": "InceptionV3", "file": "model_inception.keras"},
        ],
        "input_size": IMAGE_SIZE[0],
        "num_classes": num_classes,
        "voting": "soft",
        "accuracy": {
            "cnn": float(acc_cnn),
            "resnet": float(acc_resnet),
            "inception": float(acc_inception),
            "ensemble": float(acc_ensemble),
        }
    }

    with open(ARTIFACTS_PATH / "ensemble_config.json", "w") as f:
        json.dump(ensemble_config, f, indent=2)

    print(f"\n✅ Semua model berhasil dilatih dan disimpan di {ARTIFACTS_PATH}/")
    print("File yang dihasilkan:")
    for f_path in sorted(ARTIFACTS_PATH.iterdir()):
        size = f_path.stat().st_size
        if size > 1024 * 1024:
            print(f"  {f_path.name} ({size/1024/1024:.1f} MB)")
        else:
            print(f"  {f_path.name} ({size/1024:.1f} KB)")


if __name__ == "__main__":
    ARTIFACTS_PATH.mkdir(exist_ok=True)
    train_model()
