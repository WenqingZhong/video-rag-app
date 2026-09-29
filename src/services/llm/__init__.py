from src.config import Settings
from src.services.llm.base import ChatModel, ModelError, ModelRejected, ModelUnavailable
from src.services.llm.bedrock import BedrockChat
from src.services.llm.ollama import OllamaChat


def make_chat_model(settings: Settings, purpose: str = "text", timeout: float = 60.0) -> ChatModel:
    """purpose "text": understanding, answers, chat routing · "vision": keyframe captions."""
    if settings.llm_provider == "bedrock":
        model_id = settings.bedrock_vision_model_id if purpose == "vision" else settings.bedrock_text_model_id
        if not model_id:
            raise ValueError(f"LLM_PROVIDER=bedrock needs BEDROCK_{purpose.upper()}_MODEL_ID")
        return BedrockChat(model_id, settings.bedrock_region, timeout=timeout)
    model = settings.caption_model if purpose == "vision" else settings.understanding_model
    return OllamaChat(settings.ollama_host, model, timeout=timeout)


__all__ = [
    "BedrockChat",
    "ChatModel",
    "ModelError",
    "ModelRejected",
    "ModelUnavailable",
    "OllamaChat",
    "make_chat_model",
]
