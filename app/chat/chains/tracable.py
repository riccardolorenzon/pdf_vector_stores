from app.chat.tracing.langfuse import langfuse
from langfuse.langchain import CallbackHandler

class TracableChain: 
    def __call__(self, *args, **kwargs):
        observation = langfuse.start_observation(
            name=self.metadata["conversation_id"],
            metadata=self.metadata,
            as_type="chain"
        )
        
        handler = CallbackHandler(
            trace_context={
                "trace_id": observation.trace_id,
                "observation_id": observation.id
            }
        )
        callbacks = kwargs.get("callbacks", [])
        callbacks.append(handler)
        kwargs["callbacks"] = callbacks
        return super().__call__(*args, **kwargs)