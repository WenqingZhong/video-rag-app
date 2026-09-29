"""Models on Amazon Bedrock (production on AWS), through the Converse API.

Structured replies use tool use: one tool whose input schema is the form, and the model is required to call it,
so the reply is JSON in that shape (Bedrock's equivalent of Ollama's `format`). Credentials come from the usual
AWS chain (an IAM role on ECS/EC2, or AWS_PROFILE locally).

Not exercised against real Bedrock in this project yet: unit-tested with a stubbed client only.
"""

import json
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from src.services.llm.base import ModelRejected, ModelUnavailable
from src.services.usage import LLMCall, Operation

_RETRYABLE = {"ThrottlingException", "ServiceUnavailableException", "InternalServerException", "ModelNotReadyException"}


class BedrockChat:
    provider = "bedrock"

    def __init__(self, model_id: str, region: str, timeout: float = 60.0, client: Any = None):
        self.name = model_id
        self.client = client or boto3.client(
            "bedrock-runtime", region_name=region, config=Config(read_timeout=timeout, retries={"max_attempts": 2})
        )

    def _converse(self, **request: Any) -> dict[str, Any]:
        try:
            return self.client.converse(modelId=self.name, **request)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in _RETRYABLE:
                raise ModelUnavailable(f"Bedrock {code}: {exc}") from exc
            raise ModelRejected(f"Bedrock {code}: {exc}") from exc
        except BotoCoreError as exc:  # connection errors, timeouts
            raise ModelUnavailable(f"Bedrock unreachable: {exc}") from exc

    def _call(self, response: dict[str, Any], operation: Operation, images: int = 0) -> LLMCall:
        usage, seconds = response.get("usage", {}), response.get("metrics", {}).get("latencyMs", 0) / 1000
        return LLMCall(operation, self.name, int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0)),
                       images=images, total_sec=round(seconds, 4), output_sec=round(seconds, 4))  # fmt: skip

    def json_reply(
        self, system: str, user: str, schema: dict[str, Any], operation: Operation, max_tokens: int | None = None
    ) -> tuple[str, LLMCall]:
        response = self._converse(
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"temperature": 0, "maxTokens": max_tokens or 512},
            toolConfig={
                "tools": [
                    {"toolSpec": {"name": "reply", "description": "Reply in this structure.", "inputSchema": {"json": schema}}}
                ],
                "toolChoice": {"tool": {"name": "reply"}},
            },
        )
        content = response.get("output", {}).get("message", {}).get("content", [])
        tool_input = next((block["toolUse"]["input"] for block in content if "toolUse" in block), None)
        text = json.dumps(tool_input) if tool_input is not None else "".join(b.get("text", "") for b in content)
        return text, self._call(response, operation)

    def describe_image(self, prompt: str, image_jpeg: bytes, operation: Operation, max_tokens: int) -> tuple[str, LLMCall]:
        response = self._converse(
            messages=[
                {"role": "user", "content": [{"image": {"format": "jpeg", "source": {"bytes": image_jpeg}}}, {"text": prompt}]}
            ],
            inferenceConfig={"temperature": 0, "maxTokens": max_tokens},
        )
        content = response.get("output", {}).get("message", {}).get("content", [])
        return "".join(b.get("text", "") for b in content), self._call(response, operation, images=1)

    def health(self) -> dict[str, Any]:
        # No cheap "is this model enabled" call on bedrock-runtime: report the configuration; failures show per call.
        return {"status": "healthy", "message": f"bedrock: {self.name} (checked on first call)"}

    def close(self) -> None:
        pass
