import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", 
                                          google_api_key=os.getenv("_GOOGLE_API_KEY"))