from langchain_classic.memory import ConversationBufferMemory

from app.chat.memories.histories.sql_history import SQLChatMessageHistory

def build_sql_memory(chat_args: object) -> ConversationBufferMemory:
    """
    Build a SQL-based conversation memory for the given conversation ID.

    :param conversation_id: The unique identifier for the conversation.

    :return: An instance of ConversationBufferMemory using SQLChatMessageHistory.
    """
    message_history = SQLChatMessageHistory(conversation_id=chat_args.conversation_id)
    memory = ConversationBufferMemory(
        chat_memory=message_history,
        return_messages=True,
        memory_key="chat_history",
        output_key="answer",
    )
    return memory