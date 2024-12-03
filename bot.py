from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel
import streamlit as st
import requests
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain.docstore.document import Document
import chromadb
# Initialize chat_history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit app title
st.title("LearnPrompting Assistant")

# User input
user_input = st.text_input("Ask me anything about LearnPrompting:", placeholder="Type your question here...")

# Function to send query to the chatbot API
def get_chatbot_response(query):
    api_url = "http://localhost:8000/query"  # Correct API endpoint
    payload = {"question": query}
    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response from chatbot."), data.get("source_documents", [])
        else:
            return f"Error: {response.status_code} - {response.text}", []
    except Exception as e:
        return f"Error: {e}", []

# Handle user input
if st.button("Send") and user_input.strip():
    chatbot_response, source_docs = get_chatbot_response(user_input)
    st.session_state.chat_history.append({"user": user_input, "bot": chatbot_response})
    for doc in source_docs:
        st.session_state.chat_history.append({"source": doc})

# Display chat history
for chat in st.session_state.chat_history:
    if "user" in chat:
        st.markdown(f"**You:** {chat['user']}")
    if "bot" in chat:
        st.markdown(f"**Chatbot:** {chat['bot']}")


# Optional footer
st.markdown("---")
st.markdown("Powered by RAG and GPT-4o-mini")
