from app.chat.models import ChatArgs
from app.chat.vector_store.pinecone import build_retriever
from langchain.chains import ConversationalRetrievalChain
from app.chat.llms.chatgemini import build_llm
from app.chat.memories.sql_memory import build_sql_memory

def build_chat(chat_args: ChatArgs):
    """
    :param chat_args: ChatArgs object containing
        conversation_id, pdf_id, metadata, and streaming flag.

    :return: A chain

    Example Usage:

        chain = build_chat(chat_args)
    """

    retriever = build_retriever(chat_args)
    llm = build_llm(chat_args)
    memory = build_sql_memory(chat_args)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
