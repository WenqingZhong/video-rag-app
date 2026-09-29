import json

import httpx
import pytest

from src.services.llm import OllamaChat
from src.services.understanding import LLMIntentParser, QueryUnderstanding, find_exclusions, parse_with_rules


@pytest.mark.parametrize(
    ("query", "type_", "text", "exclude"),
    [
        ("Give me the part where the host says ‘AI is changing everything’", "quote", "AI is changing everything", []),
        ("Give me the part where the host says AI is changing everything", "quote", "AI is changing everything", []),
        ("where does he talk about the future of work?", "topic", "the future of work", []),
        ("a beach with no people", "visual", "beach", ["people"]),
        ("a street without any cars or trucks", "visual", "street", ["cars", "trucks"]),
        ("show me cars but not at night", "visual", "cars", ["night"]),
        ("give me a clip of a dog", "visual", "a dog", []),
    ],
)
def test_rule_parser(query, type_, text, exclude):
    intent = parse_with_rules(query)
    assert (intent.type, intent.text, intent.exclude) == (type_, text, exclude)


def test_find_exclusions():
    assert find_exclusions("a forest, no people and no animals") == ["people", "animals"]
    assert find_exclusions("dogs, but no puppies please") == ["puppies"]
    assert find_exclusions("a dog") == []


def understanding(reply) -> QueryUnderstanding:
    """QueryUnderstanding whose LLM returns `reply` (a dict, raw text, or an exception to raise)."""
    llm = LLMIntentParser(OllamaChat("http://ollama", "qwen2.5vl:3b"))

    def handler(request):
        if isinstance(reply, Exception):
            raise reply
        content = json.dumps(reply) if isinstance(reply, dict) else reply
        return httpx.Response(200, json={"message": {"content": content}})

    llm.chat.http = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    return QueryUnderstanding(llm)


def llm_reply(type_, phrase=None, visual=None, topic=None, exclude=()):
    return {"type": type_, "phrase": phrase, "visual": visual, "topic": topic, "exclude": list(exclude)}


def test_llm_intent_is_used_and_guessed_fields_dropped():
    reply = llm_reply("quote", phrase="AI is changing everything", visual="a host on stage", topic="AI")
    result = understanding(reply).understand("the part where the host says AI is changing everything")
    assert result.source == "llm"
    assert (result.intent.phrase, result.intent.visual, result.intent.topic) == ("AI is changing everything", None, None)


def test_rules_add_the_exclusion_the_llm_dropped():
    result = understanding(llm_reply("visual", visual="a beach")).understand("a beach with no people")
    assert result.source == "llm+rules" and result.intent.exclude == ["people"]


def test_invented_details_fall_back_to_rules():
    result = understanding(llm_reply("visual", visual="a dog sitting on a couch")).understand("give me a clip of a dog")
    assert result.source == "rules" and result.intent.visual == "a dog"
    assert "not in the request's words" in result.notes[0]


def test_invented_quote_falls_back_to_rules():
    reply = llm_reply("quote", phrase="technology will transform society")
    result = understanding(reply).understand("the part where she says AI is changing everything")
    assert result.source == "rules" and result.intent.phrase == "AI is changing everything"


def test_quote_marks_do_not_break_grounding():
    """Regression: a closing curly quote used to stick to the last word ("everything'")."""
    reply = llm_reply("quote", phrase="AI is changing everything")
    result = understanding(reply).understand("Give me the part where the host says ‘AI is changing everything’")
    assert result.source == "llm" and result.notes == []


@pytest.mark.parametrize("query", ["canine", "animals", "birds, but not in the sky"])
def test_topic_without_speech_words_becomes_visual(query):
    result = understanding(llm_reply("topic", topic=query.split(",")[0])).understand(query)
    assert result.intent.type == "visual" and "no speech word" in result.notes[0]


def test_topic_with_speech_words_is_kept():
    reply = llm_reply("topic", topic="the future of work")
    assert understanding(reply).understand("where does he talk about the future of work").intent.type == "topic"


def test_exclusions_are_grounded_and_merged():
    reply = llm_reply("visual", visual="a cat", exclude=["siamese", "dogs"])  # "dogs" was never said
    result = understanding(reply).understand("a cat, excluding siamese cats")
    assert result.intent.exclude == ["siamese cats"]


@pytest.mark.parametrize("reply", [httpx.ConnectError("ollama down"), "not json", {"type": "dance"}])
def test_llm_failures_fall_back_to_rules(reply):
    result = understanding(reply).understand("a beach with no people")
    assert result.source == "rules" and result.intent.visual == "beach" and result.intent.exclude == ["people"]


def test_llm_disabled_uses_rules():
    result = QueryUnderstanding(None).understand("give me a clip of a dog")
    assert result.source == "rules" and result.intent.visual == "a dog"


@pytest.mark.parametrize(
    ("query", "exclude"),
    [("exclude people", ["people"]), ("anything without people", ["people"]), ("nothing with boats", ["boats"])],
)
def test_rules_detect_requests_without_a_subject(query, exclude):
    intent = parse_with_rules(query)
    assert (intent.type, intent.visual, intent.exclude, intent.has_subject) == ("visual", None, exclude, False)


def test_placeholder_request_has_no_subject():
    assert not parse_with_rules("show me something").has_subject
    assert parse_with_rules("a beach with no people").has_subject


@pytest.mark.parametrize("visual", [None, "anything"])
def test_llm_no_subject_answer_is_accepted_not_rejected(visual):
    """A null/placeholder visual is a valid 'no subject' answer, not an invented one (no fallback to rules)."""
    result = understanding(llm_reply("visual", visual=visual, exclude=["people"])).understand("anything without people")
    assert result.source == "llm" and result.intent.visual is None and not result.intent.has_subject
    assert result.intent.exclude == ["people"] and result.notes[0].startswith("no subject")


def test_llm_copying_a_prompt_example_is_caught():
    """Regression: for 'exclude people' the model once answered 'a horse running' (a prompt example)."""
    result = understanding(llm_reply("visual", visual="a horse running", exclude=["people"])).understand("exclude people")
    assert result.source == "rules" and result.intent.visual is None and result.intent.exclude == ["people"]


def test_dont_want_is_a_negation():
    intent = parse_with_rules("I don't want any cars")
    assert (intent.visual, intent.exclude, intent.has_subject) == (None, ["cars"], False)
    assert parse_with_rules("a street, I do not want buses").exclude == ["buses"]


@pytest.mark.parametrize("query", ["hi", "hello there", "thanks!", "ok"])
def test_chit_chat_has_no_subject(query):
    assert not parse_with_rules(query).has_subject


def test_tells_counts_as_speech():
    """Found on the held-out set: 'tells' was missing from the speech words, so this quote became visual."""
    reply = llm_reply("quote", phrase="see you next week")
    assert understanding(reply).understand("the clip where she tells viewers see you next week").intent.type == "quote"
