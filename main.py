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

load_dotenv()

# Load a document
loader = PyPDFLoader("Learn Prompting Guide.pdf")
documents = loader.load()

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# Specify persist directory
persist_directory = './chroma_db'
# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = text_splitter.split_documents(documents)

# Define Chroma settings
chroma_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory,
)


# Store embeddings in Chroma
# Initialize Chroma with the updated API
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)


# Minimal settings
client_settings = Settings(
    persist_directory="./chroma_db"
)

client = chromadb.Client(client_settings)

print("Default tenant created successfully!")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize the LLM
llm_chain = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_chain,
    retriever=retriever,
    return_source_documents=True
)

# Testing the chain
query = "What course is free on Learn Prompting?"
response = qa_chain.invoke({"query": query})
result = response["result"]
source_documents = response["source_documents"]

print("Answer:", result)

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/query")
def query_chain(request: Query):
    query = request.question  # Extract the query from the request
    response = qa_chain.invoke({"query": query})  # Use `invoke` instead of `run`
    result = response["result"]
    source_documents = response["source_documents"]
    return {
        "response": result,
        "source_documents": [doc.page_content for doc in source_documents]  # Optional: Return document content
    }


@app.get("/")
def root():
    return {"message": "Welcome to the chatbot API. Use the /query endpoint to ask questions."}

