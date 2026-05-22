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
