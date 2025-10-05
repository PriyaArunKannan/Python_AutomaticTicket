# train.py
import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from data_prep import prepare_data, SAVE_DIR
from model import build_lstm_model, save_model
import pickle

# Directory to store model artifacts and saved models
ARTIFACT_DIR = SAVE_DIR
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_artifacts():
    """
    Load all preprocessed data and saved artifacts (tokenizer, label encoder).
    
    Returns:
        - X_train: Preprocessed training data
        - y_train: Training labels
        - X_val: Preprocessed validation data
        - y_val: Validation labels
        - X_test: Preprocessed test data
        - y_test: Test labels
        - tokenizer: Tokenizer used for tokenization during preprocessing
        - le: LabelEncoder for converting labels
    """
    X_train = np.load(os.path.join(ARTIFACT_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(ARTIFACT_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(ARTIFACT_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(ARTIFACT_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(ARTIFACT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(ARTIFACT_DIR, "y_test.npy"))
    
    # Load tokenizer and label encoder from saved files
    with open(os.path.join(ARTIFACT_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, le

def evaluate_and_report(model, X_test, y_test, le):
    """
    Evaluate the trained model on the test dataset and generate a classification report and confusion matrix.
    
    Args:
        - model: The trained model to evaluate
        - X_test: Test data for evaluation
        - y_test: Test labels
        - le: Label encoder to decode the class labels
    """
    # Generate predictions from the model
    preds = model.predict(X_test)
    preds_labels = preds.argmax(axis=1)  # Convert probabilities to class labels
    
    # Print classification report (precision, recall, F1-score, accuracy)
    print(classification_report(y_test, preds_labels, target_names=le.classes_))
    
    # Compute and plot confusion matrix
    cm = confusion_matrix(y_test, preds_labels)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")  # Save confusion matrix as an image
    print("Saved confusion_matrix.png")

def main(args):
    """
    Main function to train the LSTM model. Handles data loading, model training, 
    evaluation, and saving the trained model.
    
    Args:
        - args: Command line arguments containing hyperparameters for training
    """
    # Prepare data if not already available
    if not os.path.exists(os.path.join(ARTIFACT_DIR, "X_train.npy")):
        prepare_data()
    
    # Load the preprocessed data and artifacts
    X_train, y_train, X_val, y_val, X_test, y_test, tokenizer, le = load_artifacts()

    # Get the vocabulary size from the tokenizer (number of unique words)
    vocab_size = min(tokenizer.num_words or len(tokenizer.word_index) + 1, tokenizer.num_words)
    
    # Number of classes corresponds to the number of unique labels in the dataset
    num_classes = len(le.classes_)

    # Build the LSTM model using the provided hyperparameters
    model = build_lstm_model(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        max_len=X_train.shape[1],
        lstm_units=args.lstm_units,
        num_classes=num_classes,
        dropout=args.dropout
    )
    
    # Print the model summary for review
    model.summary()

    # Set up model callbacks for saving the best model and early stopping
    ckpt_path = os.path.join(MODEL_DIR, "best_model.h5")
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1)
    ]

    # Train the model on the training data and validate on the validation data
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)

    # Save the final trained model after training is complete
    save_model(model, os.path.join(MODEL_DIR, "final_model"))

    # Evaluate the model on the test dataset and generate reports
    evaluate_and_report(model, X_test, y_test, le)

if __name__ == "__main__":
    # Parse command line arguments to configure the training process
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--embed_dim", type=int, default=128, help="Dimensionality of the embedding layer")
    parser.add_argument("--lstm_units", type=int, default=128, help="Number of units in the LSTM layer")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for regularization")
    args = parser.parse_args()
    
    # Start the training process
    main(args)
