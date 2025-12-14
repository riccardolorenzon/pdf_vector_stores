from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory

from app.web.api import (
    get_messages_by_conversation_id, 
    add_message_to_conversation
)


class SQLChatMessageHistory(BaseChatMessageHistory, BaseModel):
    conversation_id: str

    @property
    def messages(self):
        return get_messages_by_conversation_id(conversation_id=self.conversation_id)

    def add_message(self, message) -> None:
        db_message = add_message_to_conversation(
            conversation_id=self.conversation_id,
            role=message.type,
            content=message.content,
        )
        return db_message
    
    def clear(self) -> None:
        # Not implemented
        pass   

def build_sql_memory(conversation_id: str) -> ConversationBufferMemory:
    """
    Build a SQL-based conversation memory for the given conversation ID.

    :param conversation_id: The unique identifier for the conversation.

    :return: An instance of ConversationBufferMemory using SQLChatMessageHistory.
    """
    message_history = SQLChatMessageHistory(conversation_id=conversation_id)
    memory = ConversationBufferMemory(
        chat_memory=message_history,
        return_messages=True
    )
    return memory