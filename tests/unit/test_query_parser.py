import pytest

from src.services.search import parse_query


@pytest.mark.parametrize(
    ("query", "phrase"),
    [
        ("Give me the part where the host says ‘AI is changing everything’", "AI is changing everything"),
        ("the part where she says “we are live”", "we are live"),
        ('find "the future of work"', "the future of work"),
        ("when he says 'we're back' again", "we're back"),
        ("‘I don’t know’ moment", "I don’t know"),
        ("what's the host's 'big idea' here", "big idea"),
        ("give me a clip of a dog", None),
        ("he said “”", None),
        ("he said '...'", None),
    ],
)
def test_extracts_quoted_phrase(query, phrase):
    assert parse_query(query).phrase == phrase


def test_first_quote_wins_and_full_text_kept():
    parsed = parse_query('say "one" then "two"')
    assert parsed.phrase == "one" and parsed.text == 'say "one" then "two"'
