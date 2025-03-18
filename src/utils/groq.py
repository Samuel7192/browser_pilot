from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage
)
from langchain_openai import ChatOpenAI
from typing import List, Any, Optional, Dict, Union, Mapping
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.outputs import LLMResult, ChatResult, ChatGenerationChunk, ChatGeneration
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks.manager import Callbacks

class GroqChatOpenAI(ChatOpenAI):
    """Wrapper around Groq's API to handle incompatible message formats and context limits."""
    
    def __init__(self, **kwargs):
        """Initialize the Groq Chat model with appropriate context limits."""
        # Set default max_tokens for Groq models unless explicitly provided
        model = kwargs.get("model", "llama3-8b-8192")
        
        # Configure context limits based on model
        if "mixtral-8x7b-32768" in model:
            # Mixtral has a 32k context
            max_tokens = kwargs.get("max_tokens", 30000)
        else:
            # Other models (llama3, gemma) have 8k context
            max_tokens = kwargs.get("max_tokens", 7000)
            
        kwargs["max_tokens"] = max_tokens
        super().__init__(**kwargs)
    
    def _convert_to_string_content(self, message: BaseMessage) -> BaseMessage:
        """Convert complex content to string format."""
        # If content is a list (multimodal format), extract just the text parts
        if isinstance(message.content, list):
            text_parts = []
            for item in message.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                # Skip image parts - Groq doesn't support them
            
            # Create a new message with just the text content
            if isinstance(message, HumanMessage):
                return HumanMessage(content="".join(text_parts))
            elif isinstance(message, AIMessage):
                return AIMessage(content="".join(text_parts))
            elif isinstance(message, SystemMessage):
                return SystemMessage(content="".join(text_parts))
            else:
                # For other message types, just update the content
                message.content = "".join(text_parts)
                return message
        
        return message

    def _prepare_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Process messages to ensure compatibility with Groq."""
        # First convert any multimodal messages to text-only
        processed_messages = [self._convert_to_string_content(message) for message in messages]
        
        # If we still have too many messages, keep only the most recent ones
        # Always keep the system message (first message) if present
        if len(processed_messages) > 2:
            system_message = None
            if isinstance(processed_messages[0], SystemMessage):
                system_message = processed_messages[0]
                processed_messages = processed_messages[1:]
            
            # Keep only the most recent user-assistant exchanges
            # Retaining 2-3 exchanges is usually sufficient for most tasks
            if len(processed_messages) > 4:
                processed_messages = processed_messages[-4:]
            
            # Add back the system message if it existed
            if system_message:
                processed_messages = [system_message] + processed_messages
        
        return processed_messages
    
    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Invoke the model with processed messages."""
        # Process the input to ensure compatibility
        if isinstance(input, list):
            processed_input = self._prepare_messages(input)
        else:
            processed_input = self._convert_to_string_content(input)
        
        # Call the parent class implementation with the processed input
        return super().invoke(processed_input, config, **kwargs)
    
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Async invoke with processed messages."""
        # Process the input to ensure compatibility
        if isinstance(input, list):
            processed_input = self._prepare_messages(input)
        else:
            processed_input = self._convert_to_string_content(input)
        
        # Call the parent class implementation with the processed input
        return await super().ainvoke(processed_input, config, **kwargs)