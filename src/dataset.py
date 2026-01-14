import os
import pathlib
from typing import List, Tuple

import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

def load_image_paths(
    dataset_dir: str,
    class_names: List[str]
) -> Tuple[List[str], List[int]]:
    """
    Lê os caminhos das imagens e gera os rótulos.

    Estrutura esperada:
        dataset_dir/
            Fresh/
            Spoiled/
    """
    data_dir = pathlib.Path(dataset_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"O diretório '{dataset_dir}' não existe.")

    image_paths = []
    image_labels = []

    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Pasta ausente: {class_dir}")

        files = list(class_dir.glob("*"))

        if len(files) == 0:
            raise ValueError(f"Pasta vazia ou ilegível: {class_dir}")

        image_paths.extend([str(p) for p in files])
        image_labels.extend([label] * len(files))

    return image_paths, image_labels

def stratified_split(
    paths: List[str],
    labels: List[int],
    val_size: float = 0.2,
    seed: int = 123
):
    """
    Realiza split estratificado para evitar viés de classe.
    """
    return train_test_split(
        paths,
        labels,
        test_size=val_size,
        stratify=labels,
        random_state=seed
    )

def stratified_split(
    paths: List[str],
    labels: List[int],
    val_size: float = 0.2,
    seed: int = 123
):
    """
    Realiza split estratificado para evitar viés de classe.
    """
    return train_test_split(
        paths,
        labels,
        test_size=val_size,
        stratify=labels,
        random_state=seed
    )

def load_and_preprocess_image(
    path: tf.Tensor,
    label: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Lê, decodifica e redimensiona uma imagem.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)

    return image, label

train_ds = tf.data.Dataset.from_tensor_slices(...)

def build_dataset(
    paths: List[str],
    labels: List[int],
    batch_size: int,
    shuffle: bool = False
) -> tf.data.Dataset:
    """
    Cria um tf.data.Dataset otimizado.
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)

    return ds

def create_datasets(
    dataset_dir: str,
    class_names: List[str],
    batch_size: int,
    val_size: float = 0.2
):
    """
    Pipeline completo de criação dos datasets de treino e validação.
    """
    paths, labels = load_image_paths(dataset_dir, class_names)

    X_train, X_val, y_train, y_val = stratified_split(
        paths,
        labels,
        val_size=val_size
    )

    train_ds = build_dataset(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = build_dataset(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=False
    )

    return train_ds, val_ds
