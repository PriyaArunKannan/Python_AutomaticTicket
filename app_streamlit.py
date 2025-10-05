# app_streamlit.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from infer_and_reply import load_model_artifact, load_tokenizer_labelencoder, predict_queue, generate_reply_gemini

# Set the title and layout of the Streamlit page
st.set_page_config(page_title="Ticket Classifier + Gemini Reply", layout="wide")

# App Title
st.title("🎫 Ticket Classifier — LSTM + Gemini Auto-Reply")

# Create two columns: Left column for user input and auto-reply, Right column for prediction details and chart
col1, col2 = st.columns([1, 1])

# --------------------
# Column 1: Input & Auto-Reply
# --------------------
with col1:
    st.markdown("### ✍️ Enter Ticket")
    
    # Text area for entering the support ticket
    ticket_text = st.text_area(
        "Your support ticket:", 
        height=150, 
        value="I was charged twice for my subscription this month and need a refund."
    )

    # Button to trigger prediction and auto-reply generation
    if st.button("🚀 Predict & Draft Reply", key="predict_btn"):
        """
        This function is triggered when the "Predict & Draft Reply" button is pressed. 
        It loads the necessary machine learning model and tokenizer, makes a prediction on the 
        ticket text to classify the queue, and generates a polite reply using the Gemini API. 
        The results are stored in Streamlit's session state.
        """
        # Load the tokenizer, label encoder, and model from the infer_and_reply module
        tokenizer, le = load_tokenizer_labelencoder()
        model = load_model_artifact()

        # Predict the queue category for the ticket
        idx, probs = predict_queue(model, tokenizer, ticket_text, max_len=200)
        probs = probs.flatten()  # Flatten the probability array to make it easier to handle
        queue_name = le.inverse_transform([idx])[0]  # Convert the predicted index back to queue name

        # Generate a polite and empathetic reply using Gemini API
        reply = generate_reply_gemini(ticket_text, idx, queue_name)

        # Store prediction results and generated reply in Streamlit session state for later use in col2
        st.session_state['reply'] = reply
        st.session_state['queue_name'] = queue_name
        st.session_state['top_probs'] = probs
        st.session_state['le'] = le

    # Display the auto-generated reply if available
    if 'reply' in st.session_state:
        st.markdown("### 🤖 Auto-Generated Reply")
        st.info(st.session_state['reply'], icon="🤖")

# --------------------
# Column 2: Prediction & Horizontal Chart
# --------------------
with col2:
    """
    This column displays the predicted queue name, as well as a horizontal bar chart that 
    visualizes the top-5 predicted queues and their respective probabilities. It reads 
    the prediction results stored in the session state.
    """
    # Check if the queue name and probabilities have been computed and stored in session state
    if 'queue_name' in st.session_state:
        # Display the predicted queue name
        st.success(f"✅ Predicted Queue: **{st.session_state['queue_name']}**")

        # Display a chart showing the top-5 predicted queue probabilities
        st.markdown("### 📊 Top-5 Queue Probabilities")

        # Get the indices of the top-5 predictions (sorted by highest probability)
        top_indices = np.argsort(st.session_state['top_probs'])[::-1][:5]
        top_labels = [st.session_state['le'].inverse_transform([i])[0] for i in top_indices]
        top_probs = [st.session_state['top_probs'][i] for i in top_indices]

        # Create a horizontal bar chart for the top-5 predictions
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.barh(top_labels[::-1], top_probs[::-1], color="skyblue")

        # Add probability values at the end of each bar
        for bar, prob in zip(bars, top_probs[::-1]):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{prob:.2f}", va='center')

        # Set labels and title for the chart
        ax.set_xlabel("Probability")
        ax.set_xlim(0, 1.05)  # Ensure the x-axis goes slightly beyond 1 for clarity
        ax.set_title("Top-5 Queue Predictions")

        # Display the plot
        st.pyplot(fig)
