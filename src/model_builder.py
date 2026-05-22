"""
model_builder.py

Factory functions that return compiled Keras models for the AG News
Classification project.

Two architectures are provided:
  1. A lightweight **LSTM from scratch** with an embedding layer.
  2. A **BERT-based classifier** built on top of a pre-trained transformer.

Every function accepts hyper-parameters so the notebook can experiment
without modifying source code.
"""

from typing import Optional
import tensorflow as tf
from tensorflow import keras
from transformers import TFBertModel, BertTokenizer


# ---------------------------------------------------------------------------
# 1. Scratch LSTM Model
# ---------------------------------------------------------------------------

def build_lstm_model(
    vocab_size: int,
    embedding_dim: int = 128,
    maxlen: int = 100,
    lstm_units: int = 64,
    dropout_rate: float = 0.5,
    num_classes: int = 4,
) -> keras.Model:
    """
    Build and return a compiled LSTM text-classifier.

    Architecture
    ------------
    Input -> Embedding -> SpatialDropout1D -> Bidirectional LSTM ->
    Dense(64, ReLU) -> Dropout -> Dense(num_classes, softmax)

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary (including ``<pad>`` and ``<unk>``).
    embedding_dim : int, optional
        Dimensionality of the embedding vectors (default 128).
    maxlen : int, optional
        Length of input sequences (default 100). Used only to define the
        input shape; the model itself is agnostic to this value at runtime.
    lstm_units : int, optional
        Number of hidden units in the LSTM layer (default 64).
    dropout_rate : float, optional
        Dropout rate applied after the LSTM and before the final classifier
        (default 0.5).
    num_classes : int, optional
        Number of output classes, i.e. the dimension of the softmax layer
        (default 4 for AG News).

    Returns
    -------
    keras.Model
        Compiled Keras model ready for ``model.fit()``.
    """
    model = keras.Sequential(
        [
            keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=maxlen,
                mask_zero=True,
                name="embedding",
            ),
            keras.layers.SpatialDropout1D(0.2, name="spatial_dropout"),
            keras.layers.Bidirectional(
                keras.layers.LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
                name="bi_lstm",
            ),
            keras.layers.Dense(64, activation="relu", name="dense_relu"),
            keras.layers.Dropout(dropout_rate, name="dropout"),
            keras.layers.Dense(num_classes, activation="softmax", name="output"),
        ],
        name="ag_news_lstm",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# 2. BERT-based Classifier
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
    and attaches a shallow classification head.  To fine-tune BERT layers,
    call ``model.get_layer("bert").trainable = True`` after instantiation.

    Architecture
    ------------
    Input(ids + attention_mask) -> BERT encoder -> Mean pooling over
    token embeddings -> Dense(num_classes, softmax)

    Parameters
    ----------
    bert_model_name : str, optional
        Hugging Face model identifier (default ``"bert-base-uncased"``).
    max_length : int, optional
        Maximum token length for BERT inputs (default 128).
    num_classes : int, optional
        Number of output classes (default 4 for AG News).
    learning_rate : float, optional
        Learning rate for the Adam optimiser (default 2e-5), suitable for
        fine-tuning transformer models.

    Returns
    -------
    keras.Model
        Compiled Keras model ready for ``model.fit()``.
    """
    # Load the pre-trained transformer as a Keras layer
    bert_encoder = TFBertModel.from_pretrained(bert_model_name)
    bert_encoder.trainable = False  # Freeze by default; unfreeze for fine-tuning

    # Define inputs
    input_ids = keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    attention_mask = keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_mask"
    )

    # Forward pass through BERT
    bert_outputs = bert_encoder(
        input_ids=input_ids, attention_mask=attention_mask
    )
    # bert_outputs.last_hidden_state shape: (batch, max_length, hidden_size)
    pooled = keras.layers.GlobalAveragePooling1D(name="mean_pooling")(
        bert_outputs.last_hidden_state
    )

    # Classification head
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


def get_bert_tokenizer(bert_model_name: str = "bert-base-uncased") -> BertTokenizer:
    """
    Load and return a pre-trained BERT tokenizer.

    Parameters
    ----------
    bert_model_name : str, optional
        Hugging Face model identifier (default ``"bert-base-uncased"``).

    Returns
    -------
    BertTokenizer
        Tokenizer instance with ``pad_token`` already defined.
    """
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    return tokenizer
