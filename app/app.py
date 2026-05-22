"""
app.py

Streamlit demo for the AG News Classification project.

Run with:
    streamlit run app/app.py

The app lets users type a news headline/description and see real-time
predictions from either the trained LSTM or the BERT model.
"""

import os
import sys

import numpy as np
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------------------
# Project-path setup
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

sys.path.insert(0, SRC_DIR)

from preprocessing import clean_text, texts_to_sequences, pad_sequences
from model_builder import get_bert_tokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
MAXLEN_LSTM = 100
MAXLEN_BERT = 128
VOCAB_SIZE = 20000

# ---------------------------------------------------------------------------
# Caching model loads so Streamlit does not reload on every interaction
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading LSTM model …")
def load_lstm_model():
    """Load the trained LSTM model from disk."""
    path = os.path.join(MODELS_DIR, "lstm_model.keras")
    if not os.path.exists(path):
        st.error(f"LSTM model not found at `{path}`. Train the model in the notebook first.")
        st.stop()
    return tf.keras.models.load_model(path)


@st.cache_resource(show_spinner="Loading BERT model …")
def load_bert_model():
    """Load the trained BERT classifier from disk."""
    path = os.path.join(MODELS_DIR, "bert_model.keras")
    if not os.path.exists(path):
        st.error(f"BERT model not found at `{path}`. Train the model in the notebook first.")
        st.stop()
    return tf.keras.models.load_model(path)


@st.cache_resource(show_spinner="Loading tokenizers …")
def load_tokenizers():
    """Load the LSTM vocabulary and the BERT tokenizer."""
    # LSTM vocab is not persisted as a separate file in the lean workflow,
    # so we rebuild it from the training data if available.
    # For the demo we fall back to a dummy vocab if the CSV is missing.
    vocab = {"<pad>": 0, "<unk>": 1}

    train_csv = os.path.join(PROJECT_ROOT, "data", "train.csv")
    if os.path.exists(train_csv):
        import pandas as pd
        df = pd.read_csv(train_csv, header=None, names=["class_index", "title", "description"])
        df["text"] = df["title"].astype(str) + " " + df["description"].astype(str)
        import preprocessing as prep
        cleaned = prep.clean_corpus(df["text"].tolist())
        vocab = prep.build_vocab(cleaned)

    bert_tokenizer = get_bert_tokenizer("bert-base-uncased")
    return vocab, bert_tokenizer


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="AG News Classifier", page_icon="📰")
    st.title("📰 AG News Classifier")
    st.markdown(
        "Classify news articles into **World**, **Sports**, **Business**, or **Sci/Tech** "
        "using a deep-learning model trained on the AG News dataset."
    )

    # Sidebar: model selection
    st.sidebar.header("Settings")
    model_choice = st.sidebar.radio(
        "Choose a model:",
        options=["LSTM (from scratch)", "BERT (fine-tuned)"],
        index=0,
    )

    # Text input
    user_text = st.text_area(
        "Enter a news headline or short article:",
        height=120,
        placeholder="e.g. Apple unveils new iPhone with advanced AI features …",
    )

    if st.button("Classify"):
        if not user_text.strip():
            st.warning("Please enter some text to classify.")
            return

        with st.spinner("Analysing …"):
            vocab, bert_tokenizer = load_tokenizers()

            if model_choice == "LSTM (from scratch)":
                model = load_lstm_model()
                cleaned = clean_text(user_text)
                seq = texts_to_sequences([cleaned], vocab)
                padded = pad_sequences(seq, maxlen=MAXLEN_LSTM)
                probs = model.predict(padded, verbose=0)[0]
            else:
                model = load_bert_model()
                enc = bert_tokenizer(
                    [user_text],
                    max_length=MAXLEN_BERT,
                    truncation=True,
                    padding="max_length",
                    return_tensors="np",
                )
                probs = model.predict(
                    {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]},
                    verbose=0,
                )[0]

        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        # Results
        st.success(f"**Predicted category:** {CLASS_NAMES[pred_class]} ({confidence:.2%})")

        # Probability bar chart
        st.subheader("Class Probabilities")
        probs_dict = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
        st.bar_chart(probs_dict)

    st.divider()
    st.caption(
        "Built for the DL Math college assignment. "
        "[Dataset source](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)"
    )


if __name__ == "__main__":
    main()
