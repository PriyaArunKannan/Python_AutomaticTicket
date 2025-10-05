# model.py
import os
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

def build_lstm_model(vocab_size, embed_dim=128, max_len=200, lstm_units=128, num_classes=10, dropout=0.3):
    """
    Builds and compiles a Bi-directional LSTM model for text classification.
    
    Args:
        vocab_size (int): The size of the vocabulary (number of unique words in the dataset).
        embed_dim (int, optional): The dimensionality of the embedding layer. Defaults to 128.
        max_len (int, optional): The maximum length of input sequences. Defaults to 200.
        lstm_units (int, optional): The number of units in the LSTM layer. Defaults to 128.
        num_classes (int, optional): The number of output classes (labels). Defaults to 10.
        dropout (float, optional): The dropout rate to be applied to LSTM and Dense layers. Defaults to 0.3.
    
    Returns:
        Model: A compiled Keras model ready for training.
    """
    # Input layer for sequences with shape (max_len,)
    inp = Input(shape=(max_len,), name="input")

    # Embedding layer to convert words to dense vectors
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len, name="embedding")(inp)

    # Bidirectional LSTM layer to capture context from both directions
    x = Bidirectional(LSTM(lstm_units, dropout=dropout, recurrent_dropout=0.0, return_sequences=False))(x)

    # Dropout layer to prevent overfitting
    x = Dropout(dropout)(x)

    # Output layer with softmax activation for multi-class classification
    out = Dense(num_classes, activation="softmax", name="output")(x)

    # Construct and compile the model
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    return model

def save_model(model, path):
    """
    Saves the trained Keras model to a specified file path.
    
    Args:
        model (Model): The Keras model to be saved.
        path (str): The path (including filename) where the model should be saved.
    
    Returns:
        None
    """
    # Ensure the target directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model in the specified location
    model.save(path)
    print("Model saved to", path)

def load_saved_model(path):
    """
    Loads a previously saved Keras model from the specified file path.
    
    Args:
        path (str): The path to the saved model file.
    
    Returns:
        Model: The loaded Keras model.
    """
    return load_model(path)
