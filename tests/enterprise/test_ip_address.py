"""Deterministic contract checks for ip_address signal extractor."""
import pytest
from aegislab.enterprise.ip_address import Detector, Signal, check, CATEGORY, DOMAIN


def test_category_and_domain():
    assert CATEGORY == 'ip_address'
    assert DOMAIN == 'privacy'


def test_empty_input():
    det = Detector()
    assert det.inspect("") == []
    assert det.summarize("")["count"] == 0
    assert det.summarize("")["max_confidence"] == 0


def test_invalid_type():
    with pytest.raises(TypeError):
        Detector().inspect(None)


def test_invalid_limit():
    with pytest.raises(ValueError):
        Detector(max_matches=0)


def test_custom_pattern():
    det = Detector(patterns=[r"needle"])
    result = det.inspect("Hay needle stack")
    assert len(result) == 1
    assert result[0].matched_text == "needle"
    assert result[0].start == 4
    assert result[0].end == 10
    assert 0 <= result[0].confidence <= 1
    assert result[0].category == CATEGORY


def test_case_sensitivity():
    assert len(Detector(patterns=["CASE"]).inspect("case")) == 1
    assert len(Detector(patterns=["CASE"], case_sensitive=True).inspect("case")) == 0


def test_redact_preserves_surrounding_text():
    det = Detector(patterns=[r"secret"])
    assert det.redact("my secret now") == "my [REDACTED] now"
    assert det.redact("unchanged") == "unchanged"


def test_deduplicate_identical_patterns():
    det = Detector(patterns=[r"abc", r"abc"])
    assert len(det.inspect("abc")) == 1


def test_batch_summary():
    det = Detector(patterns=["xyz"])
    summaries = det.batches(["xyz", "empty", "xyz xyz"])
    assert [row["count"] for row in summaries] == [1,0,2]


def test_max_matches():
    det = Detector(patterns=[r"abc"], max_matches=2)
    assert len(det.inspect("abc abc abc")) == 2


def test_minimum_length():
    det = Detector(patterns=[r"a"], minimum_length=2)
    assert det.inspect("aaa") == []


def test_default_factory():
    assert isinstance(Detector(), Detector)
    assert isinstance(check(""), bool)

@pytest.mark.parametrize("text, expected", [
    ("", False), ("unrelated ordinary content", False),
])
def test_negative_examples(text, expected):
    assert check(text) == expected
