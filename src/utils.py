"""
utils.py

Evaluation and visualisation helpers for the AG News Classification project.

Functions in this module generate:
  * Training-history plots (accuracy & loss curves).
  * Confusion matrices as annotated heat-maps.
  * Classification reports (precision, recall, F1).

All functions are pure (no side-effects) except for the optional ``save_path``
argument which writes a figure to disk.
"""

import os
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
)

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_training_history(
    history: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Plot model training history: accuracy and loss curves side-by-side.

    Parameters
    ----------
    history : Dict[str, Any]
        Training history object returned by ``model.fit()`` (or a plain dict
        containing keys such as ``"accuracy"``, ``"val_accuracy"``,
        ``"loss"``, ``"val_loss"``).
    save_path : str, optional
        If provided, the figure is saved to this path (e.g. ``"reports/training.png"``).
    figsize : tuple, optional
        Matplotlib figure size (default ``(12, 4)``).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Accuracy ---
    ax = axes[0]
    ax.plot(history["accuracy"], label="Train")
    if "val_accuracy" in history:
        ax.plot(history["val_accuracy"], label="Validation")
    ax.set_title("Model Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # --- Loss ---
    ax = axes[1]
    ax.plot(history["loss"], label="Train")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Validation")
    ax.set_title("Model Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    normalize: bool = False,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Plot a confusion matrix as a Seaborn heat-map.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels, shape ``(n_samples,)``.
    y_pred : np.ndarray
        Predicted integer labels, shape ``(n_samples,)``.
    class_names : list
        Ordered list of human-readable class names (e.g.
        ``["World", "Sports", "Business", "Sci/Tech"]``).
    normalize : bool, optional
        If ``True``, values are normalised row-wise to show percentages
        instead of raw counts (default ``False``).
    save_path : str, optional
        If provided, the figure is saved to disk.
    figsize : tuple, optional
        Matplotlib figure size (default ``(8, 6)``).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Avoid NaN when a row sums to zero
        fmt = ".2%"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    title = "Confusion Matrix"
    if normalize:
        title += " (Normalised)"
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------

def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    digits: int = 4,
) -> str:
    """
    Print and return a scikit-learn classification report.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    class_names : list
        Ordered list of human-readable class names.
    digits : int, optional
        Number of digits for formatting output floating-point values
        (default 4).

    Returns
    -------
    str
        The classification report as a string (suitable for writing to a log
        file or including in a PDF appendix).
    """
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=digits
    )
    print(report)
    return report


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute a concise dictionary of top-level evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.

    Returns
    -------
    Dict[str, float]
        Dictionary containing ``accuracy`` and any other aggregated metrics.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

# ---------------------------------------------------------------------------
# Hyperparameter tuning helpers
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    build_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: Dict[str, list],
    epochs: int = 10,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Run a manual grid search over hyperparameter combinations.

    Trains a fresh model for each combination in ``param_grid`` and records
    the best validation accuracy. Works for both LSTM and BERT models by
    accepting any callable that returns a compiled Keras model.

    Parameters
    ----------
    build_fn : callable
        A function that builds and returns a compiled Keras model.
        Expected signature: ``build_fn(**params) -> keras.Model``
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels (one-hot or integer).
    X_val : np.ndarray
        Validation features.
    y_val : np.ndarray
        Validation labels.
    param_grid : Dict[str, list]
        Mapping of parameter names to lists of values to try.
        Example::

            {
                "learning_rate": [1e-3, 1e-4],
                "lstm_units":    [64, 128],
            }

    epochs : int, optional
        Maximum training epochs per combination (default 10).
    batch_size : int, optional
        Batch size for each training run (default 64).

    Returns
    -------
    Dict[str, Any]
        Results dictionary. Each key is a parameter-combination string
        (e.g. ``"learning_rate=0.001_lstm_units=64"``) and each value is a
        dict with:

        * ``"history"``           - Keras History object
        * ``"best_val_accuracy"`` - float
        * ``"params"``            - dict of hyperparameter values used
    """
    import itertools

    keys         = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))
    results      = {}

    for combo in combinations:
        params   = dict(zip(keys, combo))
        run_name = "_".join(f"{k}={v}" for k, v in params.items())
        print(f"\nTraining: {run_name}")

        model = build_fn(**params)
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=0,
        )

        best_val_acc = max(history.history["val_accuracy"])
        results[run_name] = {
            "history"          : history,
            "best_val_accuracy": best_val_acc,
            "params"           : params,
        }
        print(f"  Best val_accuracy: {best_val_acc:.4f}")

    # Summary — best combination first
    print("\n=== Tuning Summary ===")
    best_run = max(results, key=lambda k: results[k]["best_val_accuracy"])
    for name, result in sorted(
        results.items(),
        key=lambda x: x[1]["best_val_accuracy"],
        reverse=True,
    ):
        marker = " <- BEST" if name == best_run else ""
        print(f"  {name}: {result['best_val_accuracy']:.4f}{marker}")

    return results


def plot_tuning_results(
    tuning_results: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot validation-accuracy curves for all hyperparameter combinations.

    Shows two charts side-by-side:

    * **Left**  - val_accuracy learning curves per combination.
    * **Right** - bar chart of best val_accuracy per combination.

    Parameters
    ----------
    tuning_results : Dict[str, Any]
        Output dictionary from :func:`tune_hyperparameters`.
    save_path : str, optional
        If provided, saves the figure to this path.
    figsize : tuple, optional
        Matplotlib figure size (default ``(12, 5)``).

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: val_accuracy curves per combination
    ax = axes[0]
    for name, result in tuning_results.items():
        val_acc = result["history"].history["val_accuracy"]
        ax.plot(val_acc, label=name)
    ax.set_title("Validation Accuracy — All Combinations")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Right: bar chart of best val_accuracy per combination
    ax    = axes[1]
    names = list(tuning_results.keys())
    accs  = [tuning_results[n]["best_val_accuracy"] for n in names]
    bars  = ax.bar(names, accs, color="steelblue")
    ax.set_title("Best Validation Accuracy per Combination")
    ax.set_xlabel("Hyperparameter Combination")
    ax.set_ylabel("Best Val Accuracy")
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{acc:.3f}",
            ha="center", va="bottom", fontsize=7,
        )
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig