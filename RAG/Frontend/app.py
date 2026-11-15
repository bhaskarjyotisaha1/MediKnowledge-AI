
import streamlit as st
import sys
import os

# Step 1: Add project root to Python path

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(project_root)

# Step 2: Import chatbot function

from chatbot import generate_response


# Step 3: Page Configuration

st.set_page_config(
    page_title="MediKnowledge AI",
    page_icon="🏥",
    layout="centered"
)


# Step 4: Application Title

st.title("🏥 MediKnowledge AI")

st.write(
    "Welcome! I am your healthcare and medicine "
    "information assistant. Ask me questions based "
    "on the available healthcare documents."
)

st.divider()


# Step 5: Initialize Chat History

if "messages" not in st.session_state:

    st.session_state.messages = []


# Step 6: Clear Chat History

if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()


# Sidebar Information

st.sidebar.title("About MediKnowledge AI")

st.sidebar.write(
    """
    This chatbot provides healthcare-related
    information using Retrieval Augmented Generation (RAG).

    Knowledge Sources:

    - Health and Hygiene
    - Classification of Drugs
    - National List of Essential Medicines (NLEM)
    """
)

st.sidebar.info(
    "For educational purposes only. "
    "Not a substitute for professional medical advice."
)


# Step 7: Display Previous Conversation

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# Step 8: Get User Input

user_question = st.chat_input(
    "Ask your healthcare question..."
)


# Step 9: Generate AI Response

if user_question:

    # Get Previous Conversation
    chat_history = st.session_state.messages.copy()

    # Display User Message
    with st.chat_message("user"):

        st.markdown(user_question)

    # Store User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # Generate AI Response
    with st.chat_message("assistant"):

        with st.spinner("Searching healthcare documents..."):

            try:

                response = generate_response(
                    user_question,
                    chat_history
                )

                st.markdown(response)

            except Exception as e:

                print(e)

                response = f"Error: {str(e)}"

                st.error(response)

    # Store AI Response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

