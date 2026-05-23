import re
import pickle
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def clean_text(text):
    """
    Clean raw text for NLP modeling.
    
    Parameters:
        text (str): Raw input string (title + description).
    
    Returns:
        str: Lowercase string containing only letters and single spaces.
    """
    # Convert to string and lowercase
    text = str(text).lower()
    # Remove anything that is not a letter or space
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_tokenizer(texts, max_words=20000, save_path='models/tokenizer.pickle'):
    """
    Fit a Keras Tokenizer on training texts and save it for later use.
    
    Parameters:
        texts (list or array): Cleaned training strings.
        max_words (int): Vocabulary cap (most frequent words only).
        save_path (str): Path to save the tokenizer pickle file.
    
    Returns:
        Tokenizer: Fitted Keras Tokenizer object.
    """
    # Create tokenizer with OOV token for unseen words
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    # Learn the vocabulary from training texts
    tokenizer.fit_on_texts(texts)
    # Ensure models/ folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save to disk so teammates and Streamlit can reuse it
    with open(save_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved. Vocabulary size: {len(tokenizer.word_index)}")
    return tokenizer


def prepare_sequences(texts, tokenizer, max_len=128):
    """
    Convert cleaned texts to padded integer sequences.
    
    Parameters:
        texts (list or array): Cleaned strings.
        tokenizer (Tokenizer): Fitted Keras Tokenizer.
        max_len (int): Pad or truncate to this many tokens.
    
    Returns:
        numpy.ndarray: Array of shape (num_samples, max_len).
    """
    # Convert words to integers using the learned vocabulary
    sequences = tokenizer.texts_to_sequences(texts)
    # Pad with zeros at the end (post padding) so LSTM sees the start first
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded
