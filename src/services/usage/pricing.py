"""What our tokens would cost at a hosted model's list price. An estimate: the model runs locally for free."""

from dataclasses import dataclass

from src.services.usage.models import LLMCall


@dataclass(frozen=True)
class Pricing:
    reference: str  # the hosted model the price comes from, shown on the dashboard
    input_per_mtok: float  # USD per million input tokens
    output_per_mtok: float  # USD per million output tokens

    def cost(self, call: LLMCall) -> float:
        return round((call.prompt_tokens * self.input_per_mtok + call.output_tokens * self.output_per_mtok) / 1e6, 8)
