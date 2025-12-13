import os 
import pinecone
from langchain_pinecone import PineconeVectorStore
from app.chat.embeddings.gemini import embeddings

pinecone_api_key = os.getenv("PINECONE_API_KEY")

vector_store = PineconeVectorStore.from_existing_index(
    index_name=os.getenv("PINECONE_INDEX_NAME"), 
    embedding=embeddings)
