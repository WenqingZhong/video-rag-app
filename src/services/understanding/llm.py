"""LLM request parsing via Ollama structured outputs: the reply is constrained to a JSON schema."""

import json

from pydantic import ValidationError

from src.services.llm import ChatModel
from src.services.understanding.intent import Intent
from src.services.usage import LLMCall

SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["quote", "visual", "topic"]},
        "phrase": {"type": ["string", "null"]},
        "visual": {"type": ["string", "null"]},
        "topic": {"type": ["string", "null"]},
        "exclude": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["type", "phrase", "visual", "topic", "exclude"],
}

# The examples deliberately differ from the evaluation requests (measure generalisation, not memorisation).
SYSTEM_PROMPT = """You convert a request for a video clip into search fields. Reply with JSON only.

type:
- "quote":  the user wants the moment someone SAYS particular words (quote marks are optional). Set phrase to exactly those words.
- "topic":  the user wants where someone TALKS ABOUT a subject, not exact words. Set topic to the subject.
- "visual": the user wants something SHOWN on screen. Set visual to what should be visible.

Rules:
- Copy the user's own words. Never add details the user did not say (no extra places, colours, actions).
- Drop request words such as "give me", "show me", "a clip of", "the part where".
- exclude: things the user says must NOT appear ("no X", "without X", "not X"). Keep them out of visual.
- Fields that don't match the type are null; exclude is [] when nothing is excluded.
- If the user does not say what they want to see or hear (only what to avoid, or just "anything"), use type "visual"
  with visual null. Never fill a field with words from these examples.

Examples:
"find the bit where the chef says don't burn the garlic" -> {"type":"quote","phrase":"don't burn the garlic","visual":null,"topic":null,"exclude":[]}
"where do they discuss climate policy" -> {"type":"topic","phrase":null,"visual":null,"topic":"climate policy","exclude":[]}
"show me a street without cars" -> {"type":"visual","phrase":null,"visual":"a street","topic":null,"exclude":["cars"]}
"I want footage of a horse running" -> {"type":"visual","phrase":null,"visual":"a horse running","topic":null,"exclude":[]}
"nothing with cars in it" -> {"type":"visual","phrase":null,"visual":null,"topic":null,"exclude":["cars"]}"""


class InvalidReply(ValueError):
    """The model answered, but not with a valid intent. Its tokens were still spent."""

    def __init__(self, message: str, call: LLMCall):
        super().__init__(message)
        self.call = call


class LLMIntentParser:
    def __init__(self, chat: ChatModel):
        self.chat = chat  # Ollama locally, Bedrock on AWS (src/services/llm)

    @property
    def model(self) -> str:
        return self.chat.name

    def parse(self, query: str) -> tuple[Intent, LLMCall]:
        text, call = self.chat.json_reply(SYSTEM_PROMPT, query, SCHEMA, "understand")
        try:
            return Intent.model_validate(json.loads(text)), call
        except (ValidationError, ValueError, KeyError) as exc:
            raise InvalidReply(f"{type(exc).__name__}: {exc}", call) from exc

    def close(self) -> None:
        self.chat.close()
