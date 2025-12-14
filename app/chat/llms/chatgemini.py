from langchain_google_genai import ChatGoogleGenerativeAI

def build_llm(chat_args):
    """
    :param chat_args: ChatArgs object containing
        conversation_id, pdf_id, metadata, and streaming flag.

    :return: A ChatGoogleGenerativeAI LLM instance

    Example Usage:

        llm = build_llm(chat_args)
    """

    return ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-lite",
        convert_system_message_to_human=True,
    )