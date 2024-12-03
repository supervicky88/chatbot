from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv


load_dotenv()

# Load a document
loader = PyPDFLoader("Learn Prompting Guide.pdf")
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
docs = text_splitter.split_documents(documents)

print(docs[0])  # Example: View a chunk of your document


# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# Convert document chunks to embeddings
#doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in docs]


# Store embeddings in Chroma
vector_store = Chroma.from_documents(docs, embeddings)

# Persist to disk if needed
#vector_store.persist()
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# Initialize the LLM
llm_chain = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)



# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm_chain,
    retriever=retriever,
    return_source_documents=True
)

# Ask a question
query = "Tell me about sander?"
response = qa_chain.invoke({"query": query})

# Extract the result and source documents
result = response["result"]
#source_documents = response["source_documents"]

print("Answer:", result)
#print("Source Documents:", source_documents)


# Define prompt templates
#summarization_prompt = PromptTemplate(input_variables=["text"], template="Summarize this: {text}")
#qa_prompt = PromptTemplate(input_variables=["summary", "question"], template="Use this summary to answer the question: {summary}\n\nQuestion: {question}")

# Combine steps into a SequentialChain
#summarize_chain = llm.bind_prompt(summarization_prompt)
#qa_chain = llm.bind_prompt(qa_prompt)

#pipeline = SequentialChain(chains=[summarize_chain, qa_chain])


app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/query")
def query_chain(request: Query):
    query = request.query  # Extract the query from the request
    response = llm_chain.invoke({
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    })
    return {"response": response}

@app.get("/")
def root():
    return {"message": "Welcome to the chatbot API. Use the /query endpoint to ask questions."}

# Run with: uvicorn main:app --reload
