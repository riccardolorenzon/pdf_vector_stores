from langchain_google_genai import ChatGoogleGenerativeAI

def build_llm(chat_args, model_name: str):
    """
    :param chat_args: ChatArgs object containing
        conversation_id, pdf_id, metadata, and streaming flag.

    :return: A ChatGoogleGenerativeAI LLM instance

    Example Usage:

        llm = build_llm(chat_args)
    """

    return ChatGoogleGenerativeAI(
        model=model_name,
        convert_system_message_to_human=True,
        streaming=chat_args.streaming,
    )