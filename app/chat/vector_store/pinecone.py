import os 
import pinecone
from langchain_pinecone import PineconeVectorStore
from app.chat.embeddings.gemini import embeddings
from app.chat.models import ChatArgs

pinecone_api_key = os.getenv("PINECONE_API_KEY")

vector_store = PineconeVectorStore.from_existing_index(
    index_name=os.getenv("PINECONE_INDEX_NAME"), 
    embedding=embeddings)

def build_retriever(chat_args: ChatArgs, k: int):
    search_kwargs = {
                    "filter": 
                        {
                            "pdf_id": chat_args.pdf_id
                        },
                    "k": k
                } if chat_args.pdf_id else {}
    return vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)