from functools import partial
from .chatgemini import build_llm

llm_map = {
    "gemini-2.5-flash-lite": partial(build_llm, model_name="gemini-2.5-flash-lite"),
    "gemini-2.5-flash": partial(build_llm, model_name="gemini-2.5-flash"),
    "gemini-2.5-pro": partial(build_llm, model_name="gemini-2.5-pro"),
}

builder = llm_map.get("gemini-2.5-flash", build_llm)