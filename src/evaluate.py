import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from dataset import create_datasets

# CONFIGURAÇÕES
DATASET_DIR = "data/meat_dataset"
CLASS_NAMES = ["Fresh", "Spoiled"]

BATCH_SIZE = 32
VAL_SPLIT = 0.2

MODEL_PATH = "models/final_meat_classifier.keras"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

print("🔹 Carregando modelo treinado...")

model = tf.keras.models.load_model(MODEL_PATH)

print("✓ Modelo carregado com sucesso")
model.summary()

print("\n🔹 Criando dataset de validação...")

_, val_ds = create_datasets(
    dataset_dir=DATASET_DIR,
    class_names=CLASS_NAMES,
    batch_size=BATCH_SIZE,
    val_size=VAL_SPLIT
)

print("✓ Dataset de validação criado")

print("\n🔹 Gerando predições...")

y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    predicted_classes = tf.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("✓ Predições geradas")

print("\n📊 RELATÓRIO DE CLASSIFICAÇÃO\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

print(report)

with open(RESULTS_DIR / "classification_report.txt", "w") as f:
    f.write(report)

print("\n🔹 Gerando matriz de confusão...")

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão")

conf_matrix_path = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(conf_matrix_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✓ Matriz de confusão salva em {conf_matrix_path}")

print("\n🔹 Avaliação quantitativa do modelo...")

loss, accuracy = model.evaluate(val_ds, verbose=0)

print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

with open(RESULTS_DIR / "evaluation_metrics.txt", "w") as f:
    f.write(f"Loss: {loss:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
