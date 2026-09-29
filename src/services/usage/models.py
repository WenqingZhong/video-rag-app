"""What one model call consumed, as reported by Ollama."""

from dataclasses import dataclass
from typing import Literal

Operation = Literal["understand", "caption", "answer", "route"]
# caption:    "ok"
# understand: "accepted" (used as is) · "adjusted" (used after the guards changed it)
#             "rejected" (thrown away: the rules answered) · "invalid" (the reply didn't parse)
# "rejected" and "invalid" are wasted tokens.
# answer:     "accepted" · "not_found" (the excerpts didn't contain it) · "rejected" (a guard failed) · "invalid"
# route:      the agent choosing an action for a chat turn: "accepted" · "rejected" (a guard sent it to the rules) · "invalid"
# any:        "cache_hit" (no call made: a cached result was used; the row records what was saved)
Outcome = Literal["ok", "accepted", "adjusted", "rejected", "invalid", "not_found", "cache_hit"]
WASTED: tuple[Outcome, ...] = ("rejected", "invalid")


@dataclass(frozen=True)
class LLMCall:
    operation: Operation
    model: str
    prompt_tokens: int  # everything the model read, image tokens included (Ollama counts it all, even when cached)
    output_tokens: int
    images: int = 0
    load_sec: float = 0.0
    prompt_sec: float = 0.0
    output_sec: float = 0.0
    total_sec: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    @classmethod
    def from_ollama(cls, body: dict, operation: Operation, model: str, images: int = 0) -> "LLMCall":
        """Ollama reports counts, and durations in nanoseconds, on every non-streamed reply."""

        def sec(key: str) -> float:
            return round((body.get(key) or 0) / 1e9, 4)

        return cls(
            operation=operation,
            model=model,
            prompt_tokens=int(body.get("prompt_eval_count") or 0),
            output_tokens=int(body.get("eval_count") or 0),
            images=images,
            load_sec=sec("load_duration"),
            prompt_sec=sec("prompt_eval_duration"),
            output_sec=sec("eval_duration"),
            total_sec=sec("total_duration"),
        )
