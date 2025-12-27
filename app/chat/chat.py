import random
from app.chat.models import ChatArgs
from app.chat.vector_store import retriever_map
from app.chat.chains.retrieval import StreamingConversationalRetrievalChain
from app.chat.llms import llm_map
from app.chat.memories import memory_map
from langchain_google_genai import ChatGoogleGenerativeAI
from app.web.api import (
    set_conversation_components,
    get_conversation_components
)
from app.chat.score import random_component_by_score
from app.chat.tracing.langfuse import langfuse
from langfuse.langchain import CallbackHandler

def select_component(
    component_type, component_map, chat_args
):
    components = get_conversation_components(
        chat_args.conversation_id
    )
    previous_component = components[component_type]

    if previous_component:
        builder = component_map[previous_component]
        return previous_component, builder(chat_args)
    else:
        random_name = random_component_by_score(component_type, component_map)
        builder = component_map[random_name]
        return random_name, builder(chat_args)
    
def build_chat(chat_args: ChatArgs):
    """
    :param chat_args: ChatArgs object containing
        conversation_id, pdf_id, metadata, and streaming flag.

    :return: A chain

    Example Usage:

        chain = build_chat(chat_args)
    """
    retriever_name, retriever = select_component(
        "retriever",
        retriever_map,
        chat_args
    )
    llm_name, llm = select_component(
        "llm",
        llm_map,
        chat_args
    )
    memory_name, memory = select_component(
        "memory",
        memory_map,
        chat_args
    )
    print(f"retriever_name = {retriever_name} \n llm_name = {llm_name} \n memory_name = {memory_name}")
    set_conversation_components(
        chat_args.conversation_id,
        llm=llm_name,
        retriever=retriever_name,
        memory=memory_name
    )
    condensed_question_llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        convert_system_message_to_human=True,
        streaming=False,
    )
    
    observation = langfuse.start_observation(
        name=chat_args.conversation_id,
        metadata=chat_args.metadata,
        as_type="chain"
    )
    
    handler = CallbackHandler(
        trace_context={
            "trace_id": observation.trace_id,
            "observation_id": observation.id
        }
    )

    return StreamingConversationalRetrievalChain.from_llm(
        llm=llm,
        condense_question_llm=condensed_question_llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda h: h,
        callbacks=[handler], 
    )
