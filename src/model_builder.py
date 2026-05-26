"""
model_builder.py

Factory functions that return compiled tf.keras models for the AG News
Classification project.

Two architectures are provided:
  1. A lightweight **BiLSTM from scratch** with an embedding layer.
  2. A **BERT-based classifier** built on top of a pre-trained transformer.

Stack
-----
This module uses classic ``tf.keras`` (TensorFlow 2.10) for native Windows
GPU support via CUDA 11.2 + cuDNN 8.1. No standalone ``keras`` package or
``KERAS_BACKEND`` env var is involved.
"""

from typing import Optional

import tensorflow as tf
from tensorflow import keras


# ---------------------------------------------------------------------------
# 0. GPU verification helper
# ---------------------------------------------------------------------------

def verify_gpu() -> dict:
    """
    Print and return TensorFlow / CUDA accelerator info.

    Returns
    -------
    dict
        Keys: ``tf_version``, ``built_with_cuda``, ``num_gpus``, ``device_name``.
    """
    gpus = tf.config.list_physical_devices("GPU")
    info = {
        "tf_version": tf.__version__,
        "built_with_cuda": tf.test.is_built_with_cuda(),
        "num_gpus": len(gpus),
        "device_name": gpus[0].name if gpus else "cpu",
    }
    # Try to grow GPU memory lazily so other processes can share the card.
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass  # Already initialised, that is fine.

    width = max(len(k) for k in info)
    for k, v in info.items():
        print(f"{k:>{width}}: {v}")
    return info


# ---------------------------------------------------------------------------
# 1. Scratch BiLSTM Model
# ---------------------------------------------------------------------------

def build_lstm_model(
    vocab_size: int,
    embedding_dim: int = 128,
    maxlen: int = 128,
    lstm_units: int = 64,
    dropout_rate: float = 0.5,
    num_classes: int = 4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Build and return a compiled BiLSTM text-classifier.

    Architecture
    ------------
    Input -> Embedding -> SpatialDropout1D -> Bidirectional LSTM(return_sequences=True)
    -> GlobalMaxPool1D -> Dense(64, ReLU) -> Dropout -> Dense(num_classes, softmax)

    GPU notes
    ---------
    ``mask_zero`` on the Embedding and ``recurrent_dropout`` on the LSTM are
    intentionally disabled so the layer runs on the cuDNN fused LSTM kernel.
    Inputs are post-padded with zeros and the downstream GlobalMaxPool is
    robust to that padding noise.

    Parameters
    ----------
    vocab_size : int
        Number of distinct token ids, including padding (0) and ``<OOV>`` (1).
    embedding_dim : int, default 128
        Dimensionality of the embedding vectors. Directly ties into the
        "embeddings as a learned linear map" topic from the course.
    maxlen : int, default 128
        Sequence length the model expects.
    lstm_units : int, default 64
        Hidden units per LSTM direction; the BiLSTM output dim is ``2 * lstm_units``.
    dropout_rate : float, default 0.5
        Dropout applied before the final classifier.
    num_classes : int, default 4
        Output classes (4 for AG News: World, Sports, Business, Sci/Tech).
    learning_rate : float, default 1e-3
        Adam learning rate.

    Returns
    -------
    keras.Model
        Compiled with ``categorical_crossentropy`` -- labels must be one-hot.
    """
    inputs = keras.layers.Input(shape=(maxlen,), dtype=tf.int32, name="input_ids")
    x = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="embedding",
    )(inputs)
    x = keras.layers.SpatialDropout1D(0.2, name="spatial_dropout")(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm_units, return_sequences=True),
        name="bi_lstm",
    )(x)
    x = keras.layers.GlobalMaxPooling1D(name="global_max_pool")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense_relu")(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="output"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ag_news_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 2. BERT-based Classifier (TF-only path, used by step 3)
# ---------------------------------------------------------------------------

def build_bert_classifier(
    bert_model_name: str = "bert-base-uncased",
    max_length: int = 128,
    num_classes: int = 4,
    learning_rate: float = 2e-5,
) -> keras.Model:
    """
    Build and return a compiled BERT-based text-classifier.

    The model loads a pre-trained BERT encoder, freezes its weights by default,
    and attaches a shallow classification head. To fine-tune BERT layers,
    set ``model.get_layer("bert").trainable = True`` after instantiation.

    Architecture
    ------------
    Input(ids + attention_mask) -> BERT encoder -> Mean pooling over token
    embeddings -> Dense(num_classes, softmax)
    """
    # Lazy import so `import model_builder` does not require transformers.
    from transformers import TFBertModel

    bert_encoder = TFBertModel.from_pretrained(bert_model_name)
    bert_encoder.trainable = False

    input_ids = keras.layers.Input(
        shape=(max_length,), dtype="int32", name="input_ids"
    )
    attention_mask = keras.layers.Input(
        shape=(max_length,), dtype="int32", name="attention_mask"
    )

    bert_outputs = bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
    pooled = keras.layers.GlobalAveragePooling1D(name="mean_pooling")(
        bert_outputs.last_hidden_state
    )
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", name="classifier"
    )(pooled)

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=outputs,
        name="ag_news_bert",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_bert_tokenizer(bert_model_name: str = "bert-base-uncased"):
    """Load and return a pre-trained BERT tokenizer (HuggingFace)."""
    from transformers import BertTokenizer
    return BertTokenizer.from_pretrained(bert_model_name)
