import streamlit as st
import logging
import time

# Import functions and modules from your code
from config import DEVICE_TYPE
from main import load_model, retriver

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the TeachGPT model
llm = load_model(DEVICE_TYPE)

# Streamlit configuration
st.title("TeachGPT Chat Interface")

# Create a container for the chat messages
chat_container = st.container()

# Create a text input for user input
user_input = chat_container.text_input("You:", "", key="user_input")

# Function to get TeachGPT response
def get_teachgpt_response(user_input):
    if user_input:
        response = retriver(DEVICE_TYPE, user_input)
        return response
    return ""

# Display user input and TeachGPT response
if user_input:
    chat_container.text("You: " + user_input)
    response = get_teachgpt_response(user_input)
    chat_container.text("TeachGPT: " + response)

# Add an option to exit the chat
if chat_container.button("Exit"):
    st.stop()

# Add custom CSS to style the chat input
st.markdown(
    """
    <style>
    .css-1v5fz9x {
        position: absolute;
        bottom: 0;
        width: 90%;
    }
    .css-10p04il {
        position: absolute;
        bottom: 20px;
        right: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
