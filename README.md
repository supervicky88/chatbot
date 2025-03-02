Project Overview
The backend for the Learn Prompting Chatbot is built using FastAPI and integrates with the OpenAI GPT-4o-mini model for language processing. The system is designed for efficient handling of user queries, seamless database operations, and robust API endpoints.

Features
Chatbot API: Handles user queries using FastAPI and integrates with OpenAI's GPT-4o-mini model.

Retrieval-Augmented Generation (RAG): Enhances chatbot responses using LangChain with customized documentation.

Database Integration: Connects to chromdb as a vector store for embeddings.

Testing and Debugging: Includes unit tests for backend components.

Dependencies
Key Libraries and Tools

FastAPI: Backend framework for building APIs.

LangChain: For language model integration and training.

PostgreSQL: Database for storing user data and feedback.

SQLAlchemy: ORM for database operations.

ChromaDB: Vector database for document retrieval.

Pytest: For unit testing backend functionality.

Refer to requirements.txt for a complete list of dependencies.

Installation and Setup
Prerequisites

Python 3.11

Environment Variables in .env:

OPENAI_API_KEY=<your_openai_api_key>

Installation
Clone the repository:git clone https://github.com/UMDMSISCapstone/learn-prompting-chatbot-python.git

Install dependencies:

pip install -r requirements.txt

Start the FastAPI server:
uvicorn main:app --reload

Access the API documentation at http://localhost:8000.

Test backend
streamlit run bot.py

Deployment
Docker:

Build and run the application using Docker for consistent deployment:

docker build -t chatbot-backend .

docker run -p 8000:8000 chatbot-backend

Cloud Platforms:

Deploy on AWS

Use managed PostgreSQL services like AWS RDS for the database.

Troubleshooting
API Errors:

Ensure the FastAPI server is running.

Check API keys and database connectivity in .env.

Model Integration:

Ensure the OpenAI API key is valid.

Debug with logs for model response issues.

Contact Information
For issues or support:

Website: https://learnprompting.org/

GitHub Issues: Submit an issue on the repository.
