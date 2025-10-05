# Automatic Ticket Classification (Many-to-One RNN) + Gemini Auto-Reply

## Overview
This project is designed to automatically classify customer support tickets into specific queues (e.g., Billing, Technical Support, General Inquiry) using a Many-to-One LSTM (Long Short-Term Memory) Recurrent Neural Network (RNN). After classifying the ticket, it generates a polite, pre-drafted acknowledgment reply using the Google Gemini API.

The system uses a simple text classification pipeline with a pre-trained tokenizer and embedding layer. The goal of this project is to create an automated process for customer support teams to quickly categorize and respond to customer inquiries.

### Key Components:
- **Text Classification**: The ticket's body is classified into predefined queues (labels) using an LSTM model.
- **Gemini Integration**: After classification, the system generates a draft reply based on the ticket content and the predicted queue using Gemini's API (replace the placeholder with actual API call details).
- **Streamlit UI**: A user-friendly interface to input tickets and view results in real-time.

## Features
- **Dataset Loading**: The dataset is sourced from the Hugging Face `Tobi-Bueck/customer-support-tickets` dataset.
- **Preprocessing**: The project performs text cleaning, tokenization, padding, and truncation using Keras utilities.
- **LSTM-based Classifier**: A Many-to-One LSTM network is used for ticket classification, providing an accurate and efficient classification pipeline.
- **Evaluation Metrics**: The training process outputs metrics like accuracy, precision, recall, F1-score, and confusion matrix for model evaluation.
- **Model Saving/Loading**: After training, the model and label encoder are saved for later inference, allowing easy reuse without retraining.
- **Gemini Integration Placeholder**: Integration with Google Gemini API for drafting acknowledgment replies (to be configured with your API key and endpoint).
- **Streamlit UI**: An interactive web interface to try out ticket classification and see the generated replies in real-time.

## Requirements

- **Python Version**: This project is developed and tested with Python 3.9+ (Python 3.10 is recommended).
- **Dependencies**: This project relies on several Python libraries for machine learning, data processing, and web UI. These dependencies are listed in `requirements.txt`.

### Installation:
To set up the project, you need to create a virtual environment, install dependencies, and set up your environment for running the project.

1. **Create a Virtual Environment**:
   Create and activate a Python virtual environment for your project:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
````

2. **Install the Dependencies**:
   Install all required libraries using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   To use the Gemini API for generating replies, you'll need to set the `GEMINI_API_KEY` environment variable. You can do this in your terminal:

   ```bash
   export GEMINI_API_KEY="your-api-key-here"  # Linux/macOS
   set GEMINI_API_KEY="your-api-key-here"     # Windows
   ```

   Alternatively, you can add this to a `.env` file and use a package like `python-dotenv` to load it.

## Quickstart

1. **Training the Model**:
   After setting up your environment, you can begin training the model. You can train the model using the following command:

   ```bash
   python train.py --epochs 6 --batch_size 64
   ```

   The model will be trained for 6 epochs (you can adjust the number of epochs and batch size as necessary). The dataset will be split into training, validation, and test sets. The model will save training progress, evaluation metrics, and artifacts in the `artifacts/` folder.

2. **Running the Streamlit UI**:
   Once the model is trained, you can test the system using the Streamlit-based UI:

   ```bash
   streamlit run app_streamlit.py
   ```

   This will start a local web server, and you can navigate to `http://localhost:8501` in your browser to test the model with different support tickets. The UI will display the predicted queue and an auto-generated reply.

## Files Overview

### `data_prep.py`

This file handles the data preprocessing pipeline. It loads the Hugging Face dataset, cleans the text, tokenizes it, and splits it into training, validation, and test sets. It also saves the tokenizer and label encoder objects to the `artifacts/` directory for future use.

### `model.py`

This file contains the function `build_lstm_model` to construct the LSTM-based neural network model for classifying tickets. It also includes functions for saving and loading the model.

### `train.py`

This file is responsible for training the LSTM model. It accepts various hyperparameters such as epochs and batch size from the command line, compiles the model, and trains it on the preprocessed data.

### `infer_and_reply.py`

This file performs ticket inference. It loads the trained model, tokenizes the ticket input, classifies it, and generates a polite reply using the Gemini API. This is where you will integrate your Gemini API key and endpoint.

### `app_streamlit.py`

This file implements the web UI using Streamlit. Users can input support tickets, get the predicted queue, and see the generated reply all in one place. It's a simple interface for interacting with the trained model.

### `utils.py`

This file contains utility functions that are used across different parts of the project. These may include helper functions for logging, saving/loading data, etc.

## Gemini API Integration

The `infer_and_reply.py` file contains a placeholder for the Gemini API integration. To use it, replace the placeholder call with the actual API client code provided by Gemini 2.5 Pro or an HTTP request.

* **Replace the Placeholder**: In the `generate_reply_gemini()` function, the HTTP request to Gemini’s API is a placeholder. You’ll need to replace it with the official Gemini API client or the correct API endpoint.

* **Set Up the API Key**: Make sure you have a valid API key from Gemini, and store it in the environment variable `GEMINI_API_KEY`.

* **Customize Reply Format**: The generated reply can be customized based on your business requirements. This placeholder sends a polite, generic reply, but you may wish to tailor it further based on ticket content.

## Notes

* **Performance Considerations**: While this scaffold uses a lightweight Keras tokenizer and embedding layer, you may consider using pretrained tokenizers or embeddings for better performance. For example, Hugging Face’s transformers (like BERT) may offer improved accuracy and handling of complex queries.

* **Gemini Integration**: Ensure that your Gemini API integration follows the proper authentication and request format as per the official documentation. Keep in mind that the API rate limits and usage costs might apply, depending on your usage level.

* **Rate-Limiting & Error Handling**: For production environments, consider adding rate-limiting, retries, and error handling to prevent failures due to API timeouts or issues.

* **Secure Storage of API Keys**: For security, avoid hardcoding your API keys directly into the code. Instead, store them in environment variables or use a secret management service.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Happy coding, and feel free to modify the project to fit your needs!

```

### Changes:
1. **Expanded Quickstart Section**: Added more detailed steps about setting up the virtual environment, installing dependencies, and running the Streamlit UI.
2. **File Overview**: Explained each file's purpose and how they contribute to the overall project.
3. **Gemini API Section**: More details on how to integrate the Gemini API, including how to replace the placeholder code and set up the API key securely.
4. **Performance Considerations and Notes**: Added advice for improving the model's performance using pretrained embeddings, as well as production tips for handling Gemini API interactions.
```
