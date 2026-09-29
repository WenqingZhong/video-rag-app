"""Ask the model to answer from the excerpts only, citing them. Ollama JSON-schema mode, temperature 0."""

import json

from pydantic import BaseModel, ValidationError

from src.services.llm import ChatModel
from src.services.understanding.llm import InvalidReply
from src.services.usage import LLMCall

SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "answer": {"type": "string"},
        "cited": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["found", "answer", "cited"],
}

SYSTEM_PROMPT = """You answer questions about videos using ONLY the numbered excerpts you are given. Reply with JSON only.

- found: true only if the excerpts contain the answer. If they don't, found is false, answer is "" and cited is [].
- answer: one or two short sentences, in your own words, using only facts from the excerpts. No outside knowledge.
- cited: the numbers of the excerpts the answer comes from."""


class ModelAnswer(BaseModel):
    found: bool
    answer: str
    cited: list[int]


class LLMAnswerer:
    def __init__(self, chat: ChatModel):
        self.chat = chat

    @property
    def model(self) -> str:
        return self.chat.name

    def answer(self, question: str, excerpts_text: str) -> tuple[ModelAnswer, LLMCall]:
        user = f"Excerpts:\n{excerpts_text}\n\nQuestion: {question}"
        text, call = self.chat.json_reply(SYSTEM_PROMPT, user, SCHEMA, "answer", max_tokens=160)
        try:
            return ModelAnswer.model_validate(json.loads(text)), call
        except (ValidationError, ValueError, KeyError) as exc:
            raise InvalidReply(f"{type(exc).__name__}: {exc}", call) from exc

    def close(self) -> None:
        self.chat.close()
