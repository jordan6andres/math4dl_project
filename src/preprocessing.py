"""
preprocessing.py

Text preprocessing utilities for the AG News Classification project.
Provides functions for cleaning raw text, tokenising, and padding sequences
so that downstream models (LSTM and BERT) receive consistently shaped inputs.
"""

import re
import numpy as np
from typing import List, Tuple, Union


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalise and clean a raw news string.

    Steps performed:
      1. Convert to lower-case.
      2. Replace URLs with the token ``<url>``.
      3. Replace e-mail addresses with the token ``<email>``.
      4. Remove characters that are not alphabetic, numeric, or basic punctuation.
      5. Collapse multiple spaces into a single space and strip leading/trailing whitespace.

    Parameters
    ----------
    text : str
        Raw input text (e.g., a news headline or description).

    Returns
    -------
    str
        Cleaned, normalised text ready for tokenisation.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "<url>", text)
    text = re.sub(r"\S+@\S+", "<email>", text)
    text = re.sub(r"[^a-z0-9.,;:!?'\"\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_corpus(corpus: List[str]) -> List[str]:
    """
    Apply :func:`clean_text` to every element in a list of strings.

    Parameters
    ----------
    corpus : List[str]
        List of raw text samples.

    Returns
    -------
    List[str]
        List of cleaned text samples in the same order.
    """
    return [clean_text(t) for t in corpus]


# ---------------------------------------------------------------------------
# Tokenisation & vocabulary helpers (for the scratch LSTM path)
# ---------------------------------------------------------------------------

def build_vocab(texts: List[str]) -> dict:
    """
    Build a word -> integer index mapping from a corpus.

    The special tokens ``<pad>`` (index 0) and ``<unk>`` (index 1) are
    reserved automatically. Words are sorted by frequency (most frequent
    gets the lowest index).

    Parameters
    ----------
    texts : List[str]
        Cleaned text samples, each a whitespace-separated string.

    Returns
    -------
    dict
        Dictionary mapping word -> integer index.
    """
    word_counts = {}
    for text in texts:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def texts_to_sequences(texts: List[str], vocab: dict) -> List[List[int]]:
    """
    Convert a list of cleaned texts into lists of integer token indices.

    Out-of-vocabulary words are mapped to the ``<unk>`` index (1).

    Parameters
    ----------
    texts : List[str]
        Cleaned text samples.
    vocab : dict
        Word -> index mapping produced by :func:`build_vocab`.

    Returns
    -------
    List[List[int]]
        List of integer sequences, one per input text.
    """
    unk_idx = vocab.get("<unk>", 1)
    sequences = []
    for text in texts:
        seq = [vocab.get(word, unk_idx) for word in text.split()]
        sequences.append(seq)
    return sequences


def pad_sequences(
    sequences: List[List[int]],
    maxlen: int,
    padding: str = "post",
    truncating: str = "post",
    value: int = 0,
) -> np.ndarray:
    """
    Pad or truncate a list of integer sequences to a fixed length.

    This is a lightweight NumPy replacement for ``keras.preprocessing.sequence.pad_sequences``
    so that the notebook can run without importing Keras during the data-prep stage.

    Parameters
    ----------
    sequences : List[List[int]]
        List of variable-length integer sequences.
    maxlen : int
        Target sequence length.
    padding : str, optional
        Where to add padding tokens — ``"pre"`` or ``"post"`` (default ``"post"``).
    truncating : str, optional
        Where to truncate if a sequence exceeds ``maxlen`` — ``"pre"`` or ``"post"``
        (default ``"post"``).
    value : int, optional
        Integer value used for padding (default 0, i.e. ``<pad>``).

    Returns
    -------
    np.ndarray
        2-D array of shape ``(len(sequences), maxlen)`` with dtype ``int32``.
    """
    padded = np.full((len(sequences), maxlen), value, dtype=np.int32)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            if truncating == "pre":
                trunc = seq[-maxlen:]
            else:
                trunc = seq[:maxlen]
        else:
            trunc = seq

        if padding == "pre":
            padded[i, -len(trunc):] = trunc
        else:
            padded[i, :len(trunc)] = trunc
    return padded


# ---------------------------------------------------------------------------
# High-level wrapper (convenience function used by the notebook)
# ---------------------------------------------------------------------------

def prepare_lstm_data(
    texts: List[str],
    maxlen: int = 100,
    vocab_size: int = 20000,
) -> Tuple[np.ndarray, dict]:
    """
    End-to-end preprocessing pipeline for the scratch LSTM model.

    Parameters
    ----------
    texts : List[str]
        Raw text samples.
    maxlen : int, optional
        Maximum sequence length after padding (default 100).
    vocab_size : int, optional
        Number of most-frequent words to keep in the vocabulary (default 20,000).
        Words outside this limit are treated as ``<unk>``.

    Returns
    -------
    Tuple[np.ndarray, dict]
        * ``padded_sequences`` — shape ``(n_samples, maxlen)``
        * ``vocab`` — the word-index dictionary
    """
    cleaned = clean_corpus(texts)
    full_vocab = build_vocab(cleaned)

    # Restrict vocabulary to top vocab_size words (keep <pad> and <unk>)
    if vocab_size > 0:
        sorted_items = sorted(full_vocab.items(), key=lambda x: x[1])
        # <pad> and <unk> are already the lowest indices; we keep the next vocab_size words
        restricted = dict(sorted_items[: vocab_size + 2])
    else:
        restricted = full_vocab

    sequences = texts_to_sequences(cleaned, restricted)
    padded = pad_sequences(sequences, maxlen=maxlen)
    return padded, restricted
