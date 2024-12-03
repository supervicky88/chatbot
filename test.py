from chromadb.config import Settings
import chromadb

client_settings = Settings(persist_directory="./chroma_db")
client = chromadb.Client(client_settings)

# Manually create the tenant if it doesn't exist
client._create_tenant_database(tenant="default_tenant")
print("Default tenant created successfully!")
