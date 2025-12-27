from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from app.chat.chains.streamable import StreamableChain
from app.chat.chains.tracable import TracableChain

class StreamingConversationalRetrievalChain(
    TracableChain, StreamableChain, ConversationalRetrievalChain
):
    pass