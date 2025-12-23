from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from app.chat.chains.streamable import StreamableChain

class StreamingConversationalRetrievalChain(
    StreamableChain, ConversationalRetrievalChain
):
    pass