import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# Konfigurasi
DATASET_PATH = Path("datasets")
ARTIFACTS_PATH = Path("artifacts")
ENSEMBLE_CONFIG_PATH = ARTIFACTS_PATH / "ensemble_config.json"
LABELS_PATH = ARTIFACTS_PATH / "labels.json"
SINGLE_MODEL_PATH = ARTIFACTS_PATH / "banana_disease_model.keras"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

def evaluate_model():
    if not LABELS_PATH.exists():
        print("Error: labels.json tidak ditemukan di artifacts/.")
        return

    # Load labels
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    # Data Generator untuk Evaluasi (tanpa augmentasi, hanya rescale)
    test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    test_generator = test_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False  # Penting untuk confusion matrix
    )

    y_true = test_generator.classes

    # === Mode Ensemble ===
    if ENSEMBLE_CONFIG_PATH.exists():
        print("Mode: ENSEMBLE (3 model + Soft Voting)")
        with open(ENSEMBLE_CONFIG_PATH, "r") as f:
            config = json.load(f)

        models = []
        model_names = []
        for model_info in config["models"]:
            model_path = ARTIFACTS_PATH / model_info["file"]
            if not model_path.exists():
                print(f"Error: Model tidak ditemukan: {model_path}")
                return
            print(f"Loading {model_info['name']} dari {model_path}...")
            models.append(tf.keras.models.load_model(model_path))
            model_names.append(model_info["name"])

        # Evaluasi masing-masing model
        all_probs = []
        individual_results = {}
        for model, name in zip(models, model_names):
            print(f"\n{'='*60}")
            print(f"Evaluasi: {name}")
            print(f"{'='*60}")

            probs = model.predict(test_generator, verbose=0)
            all_probs.append(probs)
            y_pred = np.argmax(probs, axis=1)
            acc = accuracy_score(y_true, y_pred)
            individual_results[name] = acc

            report = classification_report(y_true, y_pred, target_names=labels)
            print(report)

            # Confusion Matrix per model
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix — {name}')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            safe_name = name.lower().replace(' ', '_')
            plt.savefig(ARTIFACTS_PATH / f"confusion_matrix_{safe_name}.png", dpi=150)
            plt.close()

        # Evaluasi Ensemble (Soft Voting)
        print(f"\n{'='*60}")
        print("Evaluasi: ENSEMBLE (Soft Voting)")
        print(f"{'='*60}")

        ensemble_probs = np.mean(all_probs, axis=0)
        y_pred_ensemble = np.argmax(ensemble_probs, axis=1)
        acc_ensemble = accuracy_score(y_true, y_pred_ensemble)
        individual_results["Ensemble (Soft Voting)"] = acc_ensemble

        report_ensemble = classification_report(y_true, y_pred_ensemble, target_names=labels)
        print(report_ensemble)

        # Simpan report ensemble
        with open(ARTIFACTS_PATH / "evaluation_report.txt", "w") as f:
            f.write("Ensemble (Soft Voting) Classification Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(report_ensemble)
            f.write("\n\nPerbandingan Akurasi:\n")
            for name, acc in individual_results.items():
                marker = " <- BEST" if acc == max(individual_results.values()) else ""
                f.write(f"  {name:30s}: {acc*100:.2f}%{marker}\n")

        # Confusion Matrix Ensemble
        cm_ensemble = confusion_matrix(y_true, y_pred_ensemble)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Greens',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix — Ensemble (Soft Voting)')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(ARTIFACTS_PATH / "confusion_matrix_ensemble.png", dpi=150)
        plt.close()

        # Perbandingan akurasi
        print(f"\n{'='*60}")
        print("PERBANDINGAN AKURASI")
        print(f"{'='*60}")
        for name, acc in individual_results.items():
            marker = " <- BEST" if acc == max(individual_results.values()) else ""
            print(f"  {name:30s}: {acc*100:.2f}%{marker}")

        # Bar chart
        plt.figure(figsize=(10, 5))
        colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
        bars = plt.bar(
            individual_results.keys(),
            [v*100 for v in individual_results.values()],
            color=colors[:len(individual_results)]
        )
        plt.ylabel('Accuracy (%)')
        plt.title('Model Comparison — Validation Accuracy')
        plt.ylim(0, 100)
        for bar, val in zip(bars, individual_results.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val*100:.1f}%', ha='center', fontweight='bold')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(ARTIFACTS_PATH / "model_comparison.png", dpi=150)
        plt.close()

        print(f"\nSemua hasil evaluasi disimpan di {ARTIFACTS_PATH}/")

    # === Fallback: Single model ===
    elif SINGLE_MODEL_PATH.exists():
        print("Mode: SINGLE MODEL")
        model = tf.keras.models.load_model(SINGLE_MODEL_PATH)

        print("Mengevaluasi model pada data validasi...")
        Y_pred = model.predict(test_generator)
        y_pred = np.argmax(Y_pred, axis=1)

        print("\nClassification Report:")
        report = classification_report(y_true, y_pred, target_names=labels)
        print(report)

        with open(ARTIFACTS_PATH / "evaluation_report.txt", "w") as f:
            f.write(report)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix — Banana Disease Detection')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(ARTIFACTS_PATH / "confusion_matrix.png", dpi=150)
        plt.close()
        print(f"Confusion matrix disimpan di {ARTIFACTS_PATH / 'confusion_matrix.png'}")
    else:
        print("Error: Tidak ada model yang ditemukan. Jalankan training terlebih dahulu.")

if __name__ == "__main__":
    evaluate_model()
