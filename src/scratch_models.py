"""
scratch_models.py  –  From-scratch Keras models for AG News Classification
===========================================================================
Course  : Applied Mathematical Concepts for Deep Learning
Step    : 2 – Scratch Model

This file is intentionally self-contained.  It does NOT import from or
modify model_builder.py or preprocessing.py.

Why a separate file?
────────────────────
• model_builder.py already contains the BERT code (Step 3) and is owned
  by that teammate — we leave it untouched.
• preprocessing.py is frozen after Step 1 — we leave it untouched.
• Any incompatibilities (loss function, vocab_size offset, max_len) are
  handled here or in the notebook glue layer (see NOTEBOOK USAGE at the
  bottom of this docstring).

Two scratch architectures
──────────────────────────
  build_bilstm_model()  –  Stacked Bidirectional LSTM
  build_cnn1d_model()   –  Multi-kernel 1-D CNN (TextCNN style)

Shared utilities
────────────────
  ModelTrainer          –  OOP wrapper: train / evaluate / predict / plot
  compare_models()      –  Accuracy bar chart + val-loss curve overlay
  to_integer_labels()   –  Converts one-hot OR 1-indexed CSV labels → 0-indexed int

Class map  (0-indexed integers used throughout this file)
──────────────────────────────────────────────────────────
  0 → World   1 → Sports   2 → Business   3 → Sci/Tech

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOTEBOOK USAGE (glue between the three files)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # ── preprocessing (unchanged) ───────────────────────
    from src.preprocessing  import build_tokenizer, prepare_sequences

    tokenizer = build_tokenizer(train_texts, max_words=20_000)
    X_train   = prepare_sequences(train_texts, tokenizer, max_len=128)
    X_val     = prepare_sequences(val_texts,   tokenizer, max_len=128)
    X_test    = prepare_sequences(test_texts,  tokenizer, max_len=128)

    # AG News CSV labels are 1-indexed → shift to 0-indexed integers
    from src.scratch_models import to_integer_labels
    y_train = to_integer_labels(df_train["Class Index"].values)
    y_val   = to_integer_labels(df_val["Class Index"].values)
    y_test  = to_integer_labels(df_test["Class Index"].values)

    # ── scratch models (this file) ──────────────────────
    from src.scratch_models import (
        build_bilstm_model, build_cnn1d_model,
        ModelTrainer, compare_models,
    )

    VOCAB_SIZE = min(tokenizer.num_words, len(tokenizer.word_index))

    lstm_model = build_bilstm_model(vocab_size=VOCAB_SIZE, max_len=128)
    cnn_model  = build_cnn1d_model( vocab_size=VOCAB_SIZE, max_len=128)

    lstm_trainer = ModelTrainer(lstm_model, "BiLSTM")
    cnn_trainer  = ModelTrainer(cnn_model,  "CNN1D")

    lstm_trainer.train(X_train, y_train, X_val, y_val)
    cnn_trainer.train( X_train, y_train, X_val, y_val)

    lstm_trainer.plot_history(save_path="../reports/bilstm_curves.png")
    compare_models([lstm_trainer, cnn_trainer], X_test, y_test,
                   save_path="../reports/scratch_comparison.png")

    # ── BERT model (model_builder.py, unchanged) ────────
    # BERT uses categorical_crossentropy → needs one-hot labels
    import numpy as np
    from tensorflow.keras.utils import to_categorical
    from src.model_builder import build_bert_classifier, get_bert_tokenizer

    y_train_oh = to_categorical(y_train, num_classes=4)   # one-hot for BERT
    y_val_oh   = to_categorical(y_val,   num_classes=4)
    y_test_oh  = to_categorical(y_test,  num_classes=4)

    bert_model = build_bert_classifier()
    # ... (Step 3 teammate's code takes over here)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

NUM_CLASSES  = 4
CLASS_NAMES  = ["World", "Sports", "Business", "Sci/Tech"]

# Defaults kept in sync with preprocessing.py
_DEFAULT_VOCAB_SIZE   = 20_000   # max_words=20000 in build_tokenizer()
_DEFAULT_MAX_LEN      = 128      # max_len=128     in prepare_sequences()
_DEFAULT_EMBED_DIM    = 128
_DEFAULT_LSTM_UNITS   = 128
_DEFAULT_DROPOUT_RATE = 0.4
_DEFAULT_LR           = 1e-3
_DEFAULT_BATCH_SIZE   = 256
_DEFAULT_EPOCHS       = 20


# ──────────────────────────────────────────────────────────────────────────────
# LABEL UTILITY
# ──────────────────────────────────────────────────────────────────────────────

def to_integer_labels(raw_labels: np.ndarray) -> np.ndarray:
    """
    Normalise labels to 0-indexed integers regardless of input format.

    Handles two common cases without any changes to preprocessing.py:
      • 1-indexed integers from the AG News CSV  (1, 2, 3, 4) → (0, 1, 2, 3)
      • One-hot vectors of shape (N, 4)          → argmax → (0, 1, 2, 3)

    Parameters
    ----------
    raw_labels : np.ndarray
        Either a 1-D array of integers or a 2-D one-hot matrix.

    Returns
    -------
    np.ndarray of shape (N,) with dtype int32, values in {0, 1, 2, 3}.
    """
    labels = np.asarray(raw_labels)

    # One-hot matrix → integer indices
    if labels.ndim == 2:
        return np.argmax(labels, axis=1).astype(np.int32)

    # 1-indexed integers → 0-indexed
    if labels.min() == 1:
        return (labels - 1).astype(np.int32)

    # Already 0-indexed integers
    return labels.astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – MODEL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_bilstm_model(
    vocab_size:    int   = _DEFAULT_VOCAB_SIZE,
    embed_dim:     int   = _DEFAULT_EMBED_DIM,
    max_len:       int   = _DEFAULT_MAX_LEN,
    lstm_units:    int   = _DEFAULT_LSTM_UNITS,
    dropout_rate:  float = _DEFAULT_DROPOUT_RATE,
    num_classes:   int   = NUM_CLASSES,
    learning_rate: float = _DEFAULT_LR,
    l2_reg:        float = 1e-4,
) -> keras.Model:
    """
    Build a stacked Bidirectional-LSTM classifier from scratch.

    Architecture
    ────────────
    Embedding(vocab_size + 1, embed_dim)   ← +1 reserves index 0 for padding
      → SpatialDropout1D                   ← drops entire embedding channels
      → BiLSTM(lstm_units, return_seq=True)
      → LayerNormalization
      → Dropout
      → BiLSTM(lstm_units // 2)            ← smaller second layer
      → LayerNormalization
      → Dense(lstm_units, relu) + L2
      → BatchNormalization
      → Dropout
      → Dense(num_classes, softmax)

    Mathematical note
    ─────────────────
    The Embedding matrix E ∈ ℝ^{vocab × d} maps each token id to a dense
    vector.  The Bi-LSTM computes at each timestep t:
        h_t = LSTM_fwd(x_t, h_{t-1})  ⊕  LSTM_bwd(x_t, h_{t+1})
    capturing both left and right context.

    Loss: sparse_categorical_crossentropy — labels must be 0-indexed integers.
    Use to_integer_labels() to convert before calling model.fit().

    Note on vocab_size
    ──────────────────
    Pass the value derived from the tokenizer:
        VOCAB_SIZE = min(tokenizer.num_words, len(tokenizer.word_index))
    This file adds +1 internally for the padding index.
    """
    inputs = keras.Input(shape=(max_len,), dtype="int32", name="token_ids")

    # +1 so that padding index 0 has its own embedding slot
    x = layers.Embedding(
        input_dim=vocab_size + 1,
        output_dim=embed_dim,
        input_length=max_len,
        mask_zero=True,
        name="embedding",
    )(inputs)
    x = layers.SpatialDropout1D(dropout_rate / 2, name="spatial_dropout")(x)

    # No recurrent_dropout — keeps the CuDNN GPU kernel active
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, name="lstm_1"),
        name="bi_lstm_1",
    )(x)
    x = layers.LayerNormalization(name="layer_norm_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=False, name="lstm_2"),
        name="bi_lstm_2",
    )(x)
    x = layers.LayerNormalization(name="layer_norm_2")(x)

    x = layers.Dense(
        lstm_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_hidden",
    )(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="BiLSTM_Scratch")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn1d_model(
    vocab_size:    int   = _DEFAULT_VOCAB_SIZE,
    embed_dim:     int   = _DEFAULT_EMBED_DIM,
    max_len:       int   = _DEFAULT_MAX_LEN,
    num_filters:   int   = 128,
    kernel_sizes:  tuple = (2, 3, 5),
    dropout_rate:  float = _DEFAULT_DROPOUT_RATE,
    num_classes:   int   = NUM_CLASSES,
    learning_rate: float = _DEFAULT_LR,
) -> keras.Model:
    """
    Build a multi-kernel 1-D CNN (TextCNN) classifier from scratch.

    Architecture
    ────────────
    Embedding(vocab_size + 1, embed_dim)
      → SpatialDropout1D
      → [Conv1D(k=2) → ReLU → GlobalMaxPool] ⎫
      → [Conv1D(k=3) → ReLU → GlobalMaxPool] ⎬  concatenated
      → [Conv1D(k=5) → ReLU → GlobalMaxPool] ⎭
      → Dense(256, relu)
      → BatchNormalization
      → Dropout
      → Dense(num_classes, softmax)

    Mathematical note
    ─────────────────
    A 1-D filter w ∈ ℝ^{k × d} slides over the sequence detecting local
    n-gram patterns.  GlobalMaxPooling selects the strongest activation
    across all positions.  Parallel kernels capture uni-, bi-, and 5-gram
    signals simultaneously.

    Loss: sparse_categorical_crossentropy — labels must be 0-indexed integers.
    Use to_integer_labels() to convert before calling model.fit().
    """
    inputs = keras.Input(shape=(max_len,), dtype="int32", name="token_ids")

    # mask_zero=False — GlobalMaxPool does not support masking
    x = layers.Embedding(
        input_dim=vocab_size + 1,
        output_dim=embed_dim,
        input_length=max_len,
        mask_zero=False,
        name="embedding",
    )(inputs)
    x = layers.SpatialDropout1D(dropout_rate / 2, name="spatial_dropout")(x)

    branches = []
    for k in kernel_sizes:
        conv = layers.Conv1D(
            filters=num_filters,
            kernel_size=k,
            activation="relu",
            padding="same",
            name=f"conv1d_k{k}",
        )(x)
        pool = layers.GlobalMaxPooling1D(name=f"gmp_k{k}")(conv)
        branches.append(pool)

    merged = layers.Concatenate(name="concat")(branches)
    merged = layers.Dense(256, activation="relu", name="dense_hidden")(merged)
    merged = layers.BatchNormalization(name="batch_norm")(merged)
    merged = layers.Dropout(dropout_rate, name="dropout")(merged)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(merged)

    model = keras.Model(inputs=inputs, outputs=outputs, name="CNN1D_Scratch")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – ModelTrainer  (OOP wrapper)
# ══════════════════════════════════════════════════════════════════════════════

class ModelTrainer:
    """
    Encapsulates training, evaluation, prediction, and plotting
    for any compiled Keras classification model.

    Usage
    -----
    trainer = ModelTrainer(model, model_name="BiLSTM")
    trainer.train(X_train, y_train, X_val, y_val)
    trainer.plot_history(save_path="../reports/bilstm_curves.png")
    trainer.evaluate(X_test, y_test)
    preds  = trainer.predict(X_test)
    result = trainer.predict_single(sequence)   # ← used by Streamlit app
    """

    def __init__(
        self,
        model: keras.Model,
        model_name: str = "Model",
        checkpoint_dir: str = "models/checkpoints",
        log_dir: str = "models/logs",
    ) -> None:
        self.model      = model
        self.model_name = model_name
        self.history_   = None

        self._ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.h5")
        self._log_dir   = os.path.join(log_dir, model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(self._log_dir,  exist_ok=True)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X_train:    np.ndarray,
        y_train:    np.ndarray,
        X_val:      np.ndarray,
        y_val:      np.ndarray,
        epochs:     int = _DEFAULT_EPOCHS,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        patience:   int = 4,
    ) -> keras.callbacks.History:
        """
        Fit the model with four callbacks:
          • EarlyStopping      – stops when val_loss doesn't improve
          • ReduceLROnPlateau  – halves LR on plateau
          • ModelCheckpoint    – saves best weights  → models/checkpoints/
          • TensorBoard        – logs metrics        → models/logs/
                                 (run: tensorboard --logdir models/logs)

        y_train / y_val must be 0-indexed integer arrays.
        Use to_integer_labels() if you haven't converted them yet.
        """
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=self._ckpt_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            TensorBoard(log_dir=self._log_dir, histogram_freq=1),
        ]

        print(f"\n{'='*60}")
        print(f"  Training : {self.model_name}")
        print(f"  Train    : {len(X_train):,} samples")
        print(f"  Val      : {len(X_val):,} samples")
        print(f"  Epochs   : {epochs}   Batch: {batch_size}")
        print(f"{'='*60}\n")

        self.history_ = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
        )
        return self.history_

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, float]:
        """
        Call model.evaluate() and print a formatted summary.
        y_test must be 0-indexed integers.
        Returns {'loss': float, 'accuracy': float}.
        """
        print(f"\n[{self.model_name}] Evaluating on test set …")
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"  Test Loss     : {loss:.4f}")
        print(f"  Test Accuracy : {acc * 100:.2f}%")
        return {"loss": loss, "accuracy": acc}

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        X: np.ndarray,
        return_proba: bool = False,
    ) -> np.ndarray:
        """
        Run model.predict() on a batch.
        Returns (N,) integer labels or (N, 4) softmax matrix.
        """
        proba = self.model.predict(X, verbose=0)
        return proba if return_proba else np.argmax(proba, axis=1)

    def predict_single(self, sequence: np.ndarray) -> dict:
        """
        Predict one padded sequence.  Called by the Streamlit app (app/app.py).

        Parameters
        ----------
        sequence : 1-D int array of length max_len produced by prepare_sequences().

        Returns
        -------
        {
          'label':         str,   e.g. 'Sports'
          'class_id':      int,   e.g. 1
          'confidence':    float, e.g. 0.97
          'probabilities': {'World': 0.01, 'Sports': 0.97, ...}
        }
        """
        x    = np.expand_dims(sequence, axis=0)
        prob = self.model.predict(x, verbose=0)[0]
        idx  = int(np.argmax(prob))
        return {
            "label":         CLASS_NAMES[idx],
            "class_id":      idx,
            "confidence":    float(prob[idx]),
            "probabilities": {CLASS_NAMES[i]: float(prob[i]) for i in range(NUM_CLASSES)},
        }

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot loss and accuracy curves side-by-side.

        Parameters
        ----------
        save_path : e.g. "../reports/bilstm_curves.png"
            Directory is created automatically if it doesn't exist.
        """
        if self.history_ is None:
            raise RuntimeError("Call .train() before .plot_history().")

        h   = self.history_.history
        eps = range(1, len(h["loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"{self.model_name} – Training Curves", fontsize=14)

        axes[0].plot(eps, h["loss"],     "b-o", ms=4, label="Train Loss")
        axes[0].plot(eps, h["val_loss"], "r-o", ms=4, label="Val Loss")
        axes[0].set_title("Loss (Sparse Cat. Cross-Entropy)")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(eps, h["accuracy"],     "b-o", ms=4, label="Train Acc")
        axes[1].plot(eps, h["val_accuracy"], "r-o", ms=4, label="Val Acc")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Figure saved → {save_path}")
        plt.show()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> None:
        """Save full model (architecture + weights + optimizer state)."""
        path = path or f"models/{self.model_name}_final.h5"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"  Model saved → {path}")

    def load_best_weights(self) -> None:
        """Reload the best checkpoint written during training."""
        if os.path.exists(self._ckpt_path):
            self.model.load_weights(self._ckpt_path)
            print(f"  Best weights loaded from {self._ckpt_path}")
        else:
            print(f"  Checkpoint not found: {self._ckpt_path}")

    def summary(self) -> None:
        self.model.summary()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – COMPARISON UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def compare_models(
    trainers:  list[ModelTrainer],
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    save_path: Optional[str] = None,
) -> dict[str, dict]:
    """
    Evaluate and visually compare multiple trained ModelTrainer objects.

    Produces:
      • Bar chart of test accuracy per model
      • Overlaid validation-loss curves

    Parameters
    ----------
    trainers  : List of ModelTrainer objects (must have been trained).
    X_test    : Padded token-id array from prepare_sequences().
    y_test    : 0-indexed integer label array from to_integer_labels().
    save_path : Optional path, e.g. "../reports/scratch_comparison.png".

    Returns
    -------
    dict  model_name → {'loss': float, 'accuracy': float}
    """
    results = {t.model_name: t.evaluate(X_test, y_test) for t in trainers}

    names  = list(results.keys())
    accs   = [results[n]["accuracy"] * 100 for n in names]
    colors = ["steelblue", "darkorange", "green", "crimson"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Scratch Model Comparison", fontsize=14)

    bars = axes[0].bar(names, accs, color=colors[:len(names)])
    for bar, v in zip(bars, accs):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{v:.2f}%", ha="center", va="bottom", fontsize=11,
        )
    axes[0].set_title("Test Accuracy (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(axis="y", alpha=0.3)

    for i, t in enumerate(trainers):
        if t.history_ is not None:
            vl = t.history_.history.get("val_loss", [])
            axes[1].plot(
                range(1, len(vl) + 1), vl,
                label=t.model_name, color=colors[i % len(colors)], linewidth=2,
            )
    axes[1].set_title("Validation Loss Over Epochs")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Comparison figure saved → {save_path}")
    plt.show()

    return results