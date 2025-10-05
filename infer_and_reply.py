## File: `infer_and_reply.py`

# infer_and_reply.py
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests
import json

# Define directories for the artifacts (tokenizer, label encoder) and model files
ARTIFACT_DIR = "artifacts"
MODEL_DIR = "saved_models"

def load_tokenizer_labelencoder():
    """
    Loads the tokenizer and label encoder from pickle files.

    Returns:
        tuple: (tokenizer, label_encoder) - The loaded tokenizer and label encoder objects.
    """
    # Load the tokenizer object from a pickle file
    with open(os.path.join(ARTIFACT_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    
    # Load the label encoder object from a pickle file
    with open(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    
    return tokenizer, le

def load_model_artifact():
    """
    Loads the trained machine learning model.

    Tries to load the model from the 'saved_models' directory. It first checks if a 
    directory containing the final model is available. If not, it looks for a saved 
    .h5 model file. Raises a FileNotFoundError if neither is found.

    Returns:
        model: The loaded Keras model.
    
    Raises:
        FileNotFoundError: If no model is found.
    """
    # Define the path for the final model
    model_path_dir = os.path.join(MODEL_DIR, "final_model")
    
    # Check if the final model directory exists
    if os.path.exists(model_path_dir):
        return load_model(model_path_dir)
    
    # Check for a saved .h5 model file
    h5 = os.path.join(MODEL_DIR, "best_model.h5")
    if os.path.exists(h5):
        return load_model(h5)
    
    # Raise an error if no model is found
    raise FileNotFoundError("Model not found - train the model first.")

def predict_queue(model, tokenizer, text, max_len=200):
    """
    Predicts the category/queue for a given text using a trained model.

    This function processes the input text, tokenizes it, pads the sequences, 
    and then feeds it into the trained model for prediction.

    Args:
        model: The trained Keras model for prediction.
        tokenizer: The tokenizer to convert text into sequences.
        text (str): The input text to classify.
        max_len (int, optional): The maximum length to pad the sequences to. Defaults to 200.

    Returns:
        tuple: (label_idx, prediction_probabilities) - The index of the predicted label and the 
               probability distribution for each class.
    """
    # Clean and preprocess the input text
    text_clean = text.lower().strip()

    # Convert the text into sequences of tokens
    seq = tokenizer.texts_to_sequences([text_clean])

    # Pad the sequences to ensure consistent input size
    pad = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    # Get the model's predictions
    pred = model.predict(pad)
    
    # Extract the index of the predicted label (class with highest probability)
    label_idx = int(np.argmax(pred, axis=1)[0])

    return label_idx, pred[0]

def generate_reply_gemini(ticket_text, predicted_queue, queue_name):
    """
    Generates a reply using Gemini's generative language model API (or a fallback reply).

    This function constructs a polite and empathetic response based on the predicted 
    queue and ticket text. It calls the Gemini API for content generation.

    Args:
        ticket_text (str): The customer ticket text.
        predicted_queue (int): The predicted queue index from the model.
        queue_name (str): The name of the queue (e.g., "technical support").

    Returns:
        str: The generated or fallback response text.
    """
    # Placeholder for Gemini API key - replace with your actual API key
    api_key = "AIzaSyAJEm7Hz9S5UFakFpvoGAXcGNTi8exnuFQ"

    if not api_key:
        # Fallback reply if no API key is provided
        return f"Hello — Thank you for contacting us about your {queue_name} issue. We've received your message and will route it to our {queue_name} team. Someone will follow up shortly. — Support Team"

    # Example placeholder (Pseudo-HTTP) - replace with actual Gemini endpoint & auth
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"  # <<< REPLACE
    
    # Construct the prompt to send to the Gemini API
    prompt = (
        "You are an empathetic customer service assistant. "
        "Write a short, polite acknowledgment to the customer that: "
        f"(1) references their issue in one sentence, (2) says their ticket is routed to {queue_name}, "
        "(3) offers expected follow-up phrasing and next steps briefly. Keep tone friendly and concise.\n\n"
        f"Ticket body:\n{ticket_text}\n\nReply:"
    )

    # Prepare the body for the API request
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    # Set the API key as a query parameter and the header for JSON content
    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}

    try:
        # Send the API request to Gemini
        resp = requests.post(endpoint, headers=headers, params=params, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Extract the generated reply from the response data
        generated_text = data.get("text") or data.get("generated_text") or data.get("output", "") or data["candidates"][0]["content"]["parts"][0]["text"]
        return generated_text
    except Exception as e:
        print("Gemini call failed:", e)
        # Return a fallback response if the API call fails
        return f"Hello — Thank you for contacting us about your {queue_name} issue. We have routed your ticket to our {queue_name} team and will get back to you soon."

if __name__ == "__main__":
    """
    Main entry point for the script.
    
    Loads the tokenizer, label encoder, and trained model, then makes a prediction 
    on a sample ticket text. It uses the predicted queue to generate a reply using 
    the Gemini API and prints the results.
    """
    # Load the tokenizer and label encoder
    tokenizer, le = load_tokenizer_labelencoder()

    # Load the trained model
    model = load_model_artifact()

    # Sample customer ticket text
    sample = "My internet has been disconnecting multiple times a day. Please help!"

    # Predict the queue for the ticket
    idx, probs = predict_queue(model, tokenizer, sample)

    # Get the name of the predicted queue from the label encoder
    queue_name = le.inverse_transform([idx])[0]

    # Generate a reply using the Gemini API (or fallback)
    reply = generate_reply_gemini(sample, idx, queue_name)

    # Print the predicted queue and the generated reply
    print("Predicted queue:", queue_name)
    print("Generated reply:\n", reply)
