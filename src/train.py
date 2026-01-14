import tensorflow as tf
from pathlib import Path

from dataset import create_datasets
from model import (
    build_backbone,
    build_classifier,
    freeze_backbone_layers,
    compile_model
)

# CONFIGURAÇÕES GERAIS
DATASET_DIR = "data/meat_dataset"
CLASS_NAMES = ["Fresh", "Spoiled"]

IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 32

VAL_SPLIT = 0.2

# Treinamento
EPOCHS_PHASE_1 = 8
EPOCHS_PHASE_2 = 10

LR_PHASE_1 = 1e-4
LR_PHASE_2 = 1e-5

# Fine-tuning
TRAINABLE_LAYERS = 20

# Saídas
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

print("🔹 Criando datasets...")

train_ds, val_ds = create_datasets(
    dataset_dir=DATASET_DIR,
    class_names=CLASS_NAMES,
    batch_size=BATCH_SIZE,
    val_size=VAL_SPLIT
)

print("✓ Datasets criados com sucesso")

print("🔹 Construindo modelo (fase 1 - warm-up)...")

backbone = build_backbone(
    input_shape=IMG_SHAPE,
    trainable=False
)

model = build_classifier(
    backbone=backbone,
    num_classes=len(CLASS_NAMES)
)

compile_model(
    model,
    learning_rate=LR_PHASE_1
)

model.summary()

callbacks_phase_1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODELS_DIR / "best_model_phase1.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

# FASE 1: WARM-UP DO CLASSIFICADOR

history_phase_1 = model.fit(
    train_ds,
    epochs=EPOCHS_PHASE_1,
    validation_data=val_ds,
    callbacks=callbacks_phase_1,
    verbose=1
)

print("✓ Warm-up concluído")

# FASE 2: FINE-TUNING DO BACKBONE

freeze_backbone_layers(
    backbone,
    num_trainable_layers=TRAINABLE_LAYERS
)

compile_model(
    model,
    learning_rate=LR_PHASE_2
)

callbacks_phase_2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=MODELS_DIR / "best_model_phase2.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

history_phase_2 = model.fit(
    train_ds,
    epochs=EPOCHS_PHASE_1 + EPOCHS_PHASE_2,
    initial_epoch=EPOCHS_PHASE_1,
    validation_data=val_ds,
    callbacks=callbacks_phase_2,
    verbose=1
)

print("✓ Fine-tuning concluído")

final_model_path = MODELS_DIR / "final_meat_classifier.keras"
model.save(final_model_path)

print(f"\n✓ Modelo final salvo em: {final_model_path}")
