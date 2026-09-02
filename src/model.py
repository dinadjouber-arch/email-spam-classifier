
import tensorflow as tf
from . import config


def build_model(vocab_size: int) -> tf.keras.Model:
    """
    Construit et compile le modèle LSTM de classification binaire (spam / ham).

    la couche de sortie utilise une activation 'sigmoid', donc la loss
    doit être configurée avec from_logits=False (le modèle sort déjà des
    probabilités, pas des logits bruts).
    """
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=config.EMBEDDING_DIM,
            mask_zero=True,
            ),

            tf.keras.layers.LSTM(config.LSTM_UNITS),
            tf.keras.layers.Dense(config.DENSE_UNITS, activation="relu"),
            tf.keras.layers.Dropout(config.DROPOUT_RATE),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model
