from langchain_classic.memory import ConversationBufferWindowMemory
from app.chat.memories.histories.sql_history import SQLChatMessageHistory

def build_window_buffer_memory(chat_args: object) -> ConversationBufferWindowMemory:
    """
    Build a window-based conversation memory for the given conversation ID.

    :param conversation_id: The unique identifier for the conversation.

    :return: An instance of ConversationBufferWindowMemory using SQLChatMessageHistory.
    """
    message_history = SQLChatMessageHistory(conversation_id=chat_args.conversation_id)
    memory = ConversationBufferWindowMemory(
        chat_memory=message_history,
        return_messages=True,
        memory_key="chat_history",
        output_key="answer",
        k=2,  # Number of recent messages to retain
    )
    return memory