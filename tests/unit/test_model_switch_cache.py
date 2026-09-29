from src.config import Settings
from src.services.answering.cache import AnswerCache
from src.services.cache import CacheClient, IndexVersion
from tests.unit.test_caching import FakeRedis


def test_switching_to_bedrock_starts_a_new_answer_cache():
    cache = CacheClient(FakeRedis())
    ollama = AnswerCache(cache, IndexVersion(cache), Settings(_env_file=None))
    bedrock = AnswerCache(cache, IndexVersion(cache), Settings(_env_file=None, llm_provider="bedrock", bedrock_text_model_id="m"))
    assert ollama.key("a dog", None, None, 1) != bedrock.key("a dog", None, None, 1)
