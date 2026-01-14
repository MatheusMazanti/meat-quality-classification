import tensorflow as tf

def build_backbone(
    input_shape=(224, 224, 3),
    trainable: bool = False
) -> tf.keras.Model:
    """
    Cria o backbone MobileNetV2 pré-treinado na ImageNet.

    Args:
        input_shape: Dimensão da imagem de entrada.
        trainable: Define se o backbone será treinável.

    Returns:
        Modelo base MobileNetV2.
    """
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    backbone.trainable = trainable
    return backbone

def build_classifier(
    backbone: tf.keras.Model,
    num_classes: int = 2
) -> tf.keras.Model:
    """
    Constrói o modelo completo com backbone + head customizada.
    """
    inputs = tf.keras.Input(shape=backbone.input_shape[1:])

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = backbone(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def freeze_backbone_layers(
    backbone: tf.keras.Model,
    num_trainable_layers: int = 20
):
    """
    Congela todas as camadas do backbone,
    exceto as últimas N.
    """
    total_layers = len(backbone.layers)
    freeze_until = total_layers - num_trainable_layers

    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False

    for layer in backbone.layers[freeze_until:]:
        layer.trainable = True

def compile_model(
    model: tf.keras.Model,
    learning_rate: float
):
    """
    Compila o modelo com SparseCategoricalCrossentropy.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        ),
        metrics=["accuracy"]
    )