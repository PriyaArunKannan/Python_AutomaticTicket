# data_prep.py

import os
import json
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Directory where artifacts will be saved
SAVE_DIR = "artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)

def load_hf_dataset():
    """
    Loads the 'customer-support-tickets' dataset from Hugging Face.
    Converts the dataset into a Pandas DataFrame and returns only the 'body' and 'queue' columns.
    
    Returns:
        pd.DataFrame: A DataFrame containing the 'body' and 'queue' columns.
    """
    # Load dataset using Hugging Face datasets library
    ds = load_dataset("Tobi-Bueck/customer-support-tickets")
    # Extract the 'train' split and convert to DataFrame
    df = pd.DataFrame(ds['train'])
    # Keep only the 'body' (ticket content) and 'queue' (ticket category)
    df = df[['body', 'queue']].dropna().reset_index(drop=True)
    return df

def clean_text(text):
    """
    Cleans the input text by:
    - Lowercasing the text.
    - Stripping leading/trailing spaces.
    - Collapsing multiple spaces into one.
    
    Args:
        text (str): The raw input text.
    
    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    # Basic cleaning steps
    text = text.lower().strip()
    text = " ".join(text.split())  # Collapse multiple whitespaces into one
    return text

def prepare_tokenizer(texts, num_words=20000, oov_token="<OOV>"):
    """
    Prepares a tokenizer that converts text into sequences of integers.
    
    Args:
        texts (list): A list of text documents to fit the tokenizer.
        num_words (int, optional): The maximum number of words to keep. Defaults to 20000.
        oov_token (str, optional): Token to use for out-of-vocabulary words. Defaults to "<OOV>".
    
    Returns:
        Tokenizer: A Keras Tokenizer object fitted on the provided texts.
    """
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def prepare_data(max_len=200, num_words=20000, test_size=0.15, val_size=0.1, random_state=42):
    """
    Prepares the data for training:
    - Loads the dataset.
    - Cleans the text.
    - Encodes the labels.
    - Tokenizes the text.
    - Pads the sequences.
    - Splits the data into training, validation, and test sets.
    - Saves the processed data and artifacts (tokenizer, label encoder) to disk.

    Args:
        max_len (int, optional): The maximum length of the sequences after padding. Defaults to 200.
        num_words (int, optional): The number of words to keep in the tokenizer. Defaults to 20000.
        test_size (float, optional): The proportion of the dataset to use for testing. Defaults to 0.15.
        val_size (float, optional): The proportion of the dataset to use for validation. Defaults to 0.1.
        random_state (int, optional): The random seed for reproducibility. Defaults to 42.
    
    Returns:
        dict: A dictionary containing the processed data and the tokenizer, label encoder.
    """
    # Load and clean the dataset
    df = load_hf_dataset()
    df['body_clean'] = df['body'].apply(clean_text)

    # Encode labels (queues)
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['queue'])

    X = df['body_clean'].values  # Features: cleaned text
    y = df['label'].values  # Labels: encoded queue names

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y)
    
    # Calculate the validation proportion relative to the temporary split (for val/test)
    val_prop = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - val_prop), random_state=random_state, stratify=y_temp)

    # Tokenizer should be fit only on the training data
    tokenizer = prepare_tokenizer(X_train, num_words=num_words)

    # Convert text to sequences (numerical representation)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq = tokenizer.texts_to_sequences(X_val)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Pad the sequences to ensure uniform length
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    # Save the processed data and artifacts (tokenizer, label encoder)
    artifacts = {
        "tokenizer": tokenizer,
        "label_encoder": le,
        "X_train": X_train_pad, "y_train": y_train,
        "X_val": X_val_pad, "y_val": y_val,
        "X_test": X_test_pad, "y_test": y_test
    }

    # Save tokenizer and label encoder to disk
    with open(os.path.join(SAVE_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(SAVE_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    # Save the padded sequences and labels as numpy arrays
    np.save(os.path.join(SAVE_DIR, "X_train.npy"), X_train_pad)
    np.save(os.path.join(SAVE_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(SAVE_DIR, "X_val.npy"), X_val_pad)
    np.save(os.path.join(SAVE_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(SAVE_DIR, "X_test.npy"), X_test_pad)
    np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)

    print("Saved artifacts to", SAVE_DIR)
    return artifacts

if __name__ == "__main__":
    """
    Main entry point for preparing the data. It runs the `prepare_data` function to load, clean, tokenize, 
    and save the dataset and necessary artifacts for model training.
    """
    prepare_data()
