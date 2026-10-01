import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from src.config import Settings
from src.services.captioning import Captioner, CaptionError
from src.services.llm import AnthropicChat, BedrockChat, ModelRejected, ModelUnavailable, OllamaChat, make_chat_model
from src.services.understanding import LLMIntentParser
from tests.unit.test_index_lifecycle import frame

SCHEMA = {"type": "object", "properties": {"type": {"type": "string"}}, "required": ["type"]}


# ---- Ollama --------------------------------------------------------------------------------------------------------
def ollama(handler) -> OllamaChat:
    chat = OllamaChat("http://ollama", "qwen2.5vl:3b")
    chat.http = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return chat


def test_ollama_sends_the_schema_and_counts_tokens():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"type": "visual"}'}, "prompt_eval_count": 400, "eval_count": 9})

    text, call = ollama(handler).json_reply("system", "a dog", SCHEMA, "understand", max_tokens=60)
    assert json.loads(text) == {"type": "visual"} and (call.prompt_tokens, call.output_tokens) == (400, 9)
    assert seen["format"] == SCHEMA and seen["options"] == {"temperature": 0, "num_predict": 60}


@pytest.mark.parametrize(("status", "error"), [(503, ModelUnavailable), (400, ModelRejected)])
def test_ollama_errors_say_whether_to_retry(status, error):
    with pytest.raises(error):
        ollama(lambda r: httpx.Response(status, text="nope")).json_reply("s", "u", SCHEMA, "understand")


def test_ollama_down_is_unavailable():
    def refuse(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ModelUnavailable):
        ollama(refuse).describe_image("Describe.", b"jpeg", "caption", 80)


# ---- Bedrock -------------------------------------------------------------------------------------------------------
def converse_reply(content: list[dict], tokens=(350, 20)) -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "usage": {"inputTokens": tokens[0], "outputTokens": tokens[1]},
        "metrics": {"latencyMs": 820},
    }


def test_bedrock_forces_a_tool_call_to_get_the_schema():
    client = MagicMock()
    client.converse.return_value = converse_reply([{"toolUse": {"toolUseId": "t1", "name": "reply", "input": {"type": "quote"}}}])
    chat = BedrockChat("some.model-id", "us-east-1", client=client)

    text, call = chat.json_reply("system", "the part where he says hi", SCHEMA, "understand", max_tokens=60)

    assert json.loads(text) == {"type": "quote"}
    assert (call.prompt_tokens, call.output_tokens, call.total_sec, call.model) == (350, 20, 0.82, "some.model-id")
    request = client.converse.call_args.kwargs
    assert request["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"] == SCHEMA
    assert request["toolConfig"]["toolChoice"] == {"tool": {"name": "reply"}}
    assert request["inferenceConfig"] == {"temperature": 0, "maxTokens": 60}


def test_bedrock_image_is_sent_as_bytes():
    client = MagicMock()
    client.converse.return_value = converse_reply([{"text": "Two dogs on grass."}], tokens=(1200, 8))
    text, call = BedrockChat("vision.model", "us-east-1", client=client).describe_image("Describe.", b"jpeg", "caption", 80)
    assert text == "Two dogs on grass." and call.images == 1
    image_block = client.converse.call_args.kwargs["messages"][0]["content"][0]
    assert image_block == {"image": {"format": "jpeg", "source": {"bytes": b"jpeg"}}}


@pytest.mark.parametrize(("code", "error"), [("ThrottlingException", ModelUnavailable), ("AccessDeniedException", ModelRejected)])
def test_bedrock_errors_say_whether_to_retry(code, error):
    client = MagicMock()
    client.converse.side_effect = ClientError({"Error": {"Code": code, "Message": "x"}}, "Converse")
    with pytest.raises(error):
        BedrockChat("m", "us-east-1", client=client).json_reply("s", "u", SCHEMA, "route")


def test_bedrock_unreachable_is_unavailable():
    client = MagicMock()
    client.converse.side_effect = EndpointConnectionError(endpoint_url="https://bedrock-runtime")
    with pytest.raises(ModelUnavailable):
        BedrockChat("m", "us-east-1", client=client).json_reply("s", "u", SCHEMA, "route")


# ---- callers don't care which ---------------------------------------------------------------------------------------
def test_the_same_parser_works_on_either_provider():
    client = MagicMock()
    intent_json = {"type": "visual", "phrase": None, "visual": "a dog", "topic": None, "exclude": []}
    client.converse.return_value = converse_reply([{"toolUse": {"toolUseId": "t", "name": "reply", "input": intent_json}}])
    intent, call = LLMIntentParser(BedrockChat("m", "us-east-1", client=client)).parse("a dog")
    assert intent.visual == "a dog" and call.operation == "understand"


def test_a_rejected_image_is_a_caption_error_but_unavailable_is_retried():
    rejected = MagicMock()
    rejected.describe_image.side_effect = ModelRejected("bad image")
    with pytest.raises(CaptionError):
        Captioner(rejected, prompt="Describe.").caption_with_usage(frame())
    down = MagicMock()
    down.describe_image.side_effect = ModelUnavailable("busy")
    with pytest.raises(ModelUnavailable):  # the worker retries these (TRANSIENT_ERRORS)
        Captioner(down, prompt="Describe.").caption_with_usage(frame())


def test_provider_is_a_setting():
    assert isinstance(make_chat_model(Settings(_env_file=None)), OllamaChat)
    with pytest.raises(ValueError, match="BEDROCK_TEXT_MODEL_ID"):
        make_chat_model(Settings(_env_file=None, llm_provider="bedrock"))
    bedrock = make_chat_model(Settings(_env_file=None, llm_provider="bedrock", bedrock_text_model_id="m"), "text")
    assert isinstance(bedrock, BedrockChat) and bedrock.name == "m"


# ---- Anthropic API --------------------------------------------------------------------------------------------------
def anthropic_response(blocks, input_tokens=420, output_tokens=25):
    return SimpleNamespace(content=blocks, usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens))


def test_anthropic_forces_a_tool_call_to_get_the_schema():
    client = MagicMock()
    client.messages.create.return_value = anthropic_response(
        [SimpleNamespace(type="tool_use", input={"kind": "visual", "text": "a dog"})]
    )
    text, call = AnthropicChat("claude-haiku-4-5-20251001", "key", client=client).json_reply(
        "sys", "a dog", SCHEMA, "understand", 60
    )
    request = client.messages.create.call_args.kwargs
    assert request["tool_choice"] == {"type": "tool", "name": "reply"} and request["tools"][0]["input_schema"] == SCHEMA
    assert request["system"] == "sys" and request["max_tokens"] == 60 and request["model"] == "claude-haiku-4-5-20251001"
    assert json.loads(text) == {"kind": "visual", "text": "a dog"}
    assert (call.prompt_tokens, call.output_tokens, call.operation) == (420, 25, "understand")


def test_anthropic_image_is_sent_as_base64():
    client = MagicMock()
    client.messages.create.return_value = anthropic_response([SimpleNamespace(type="text", text="A dog.")], 300, 8)
    text, call = AnthropicChat("m", "key", client=client).describe_image("Describe.", b"jpeg", "caption", 80)
    image = client.messages.create.call_args.kwargs["messages"][0]["content"][0]
    assert image["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "anBlZw=="}
    assert text == "A dog." and call.images == 1


def _status_error(cls, code):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("error", response=httpx.Response(code, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (lambda: _status_error(anthropic.RateLimitError, 429), ModelUnavailable),
        (lambda: _status_error(anthropic.InternalServerError, 500), ModelUnavailable),
        (lambda: anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com")), ModelUnavailable),
        (lambda: _status_error(anthropic.AuthenticationError, 401), ModelRejected),
        (lambda: _status_error(anthropic.BadRequestError, 400), ModelRejected),
    ],
)
def test_anthropic_errors_say_whether_to_retry(error, expected):
    client = MagicMock()
    client.messages.create.side_effect = error()
    with pytest.raises(expected):
        AnthropicChat("m", "key", client=client).json_reply("s", "u", SCHEMA, "route")


def test_anthropic_needs_a_key():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        make_chat_model(Settings(_env_file=None, llm_provider="anthropic"))
    chat = make_chat_model(Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="k"), "vision")
    assert isinstance(chat, AnthropicChat) and chat.name == "claude-haiku-4-5-20251001"
