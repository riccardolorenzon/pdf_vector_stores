from app.web.api import add_message_to_conversation, get_messages_by_conversation_id
from langchain_classic.schema import BaseChatMessageHistory

class SQLChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id

    @property
    def messages(self):
        return get_messages_by_conversation_id(
            conversation_id=self.conversation_id)

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