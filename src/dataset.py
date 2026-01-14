import os
import pathlib
from typing import Tuple, List

import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess_image(
    path: tf.Tensor,
    label: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Lê uma imagem do disco, decodifica, redimensiona e retorna com o rótulo.

    Args:
        path: Caminho do arquivo da imagem.
        label: Rótulo inteiro da classe.

    Returns:
        image: Tensor float32 no formato (224, 224, 3)
        label: Tensor inteiro
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

    Args:
        dataset_dir: Caminho base do dataset.
        class_names: Lista ordenada de nomes das classes.

    Returns:
        image_paths: Lista de caminhos das imagens.
        image_labels: Lista de rótulos inteiros.
    """
    data_dir = pathlib.Path(dataset_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {dataset_dir}")

    image_paths = []
    image_labels = []

    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Pasta ausente: {class_dir}")

        files = list(class_dir.glob("*"))
        if len(files) == 0:
            raise ValueError(f"Pasta vazia: {class_dir}")

        image_paths.extend([str(p) for p in files])
        image_labels.extend([label] * len(files))

    return image_paths, image_labels

def split_dataset(
    image_paths: List[str],
    image_labels: List[int],
    val_size: float = 0.2,
    seed: int = 42
):
    """
    Realiza split estratificado em treino e validação.
    """
    return train_test_split(
        image_paths,
        image_labels,
        test_size=val_size,
        stratify=image_labels,
        random_state=seed
    )

def build_tf_dataset(
    paths: List[str],
    labels: List[int],
    batch_size: int,
    shuffle: bool = False
) -> tf.data.Dataset:
    """
    Constrói um tf.data.Dataset otimizado.

    Args:
        paths: Lista de caminhos das imagens.
        labels: Lista de rótulos.
        batch_size: Tamanho do batch.
        shuffle: Se deve embaralhar.

    Returns:
        tf.data.Dataset
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

    X_train, X_val, y_train, y_val = split_dataset(
        paths,
        labels,
        val_size=val_size
    )

    train_ds = build_tf_dataset(
        X_train,
        y_train,
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = build_tf_dataset(
        X_val,
        y_val,
        batch_size=batch_size,
        shuffle=False
    )

    return train_ds, val_ds

