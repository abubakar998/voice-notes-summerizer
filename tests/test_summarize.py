"""Map-reduce splitting, transient-failure retry, and provider fallback.

No network: providers are replaced with stand-ins that fail on cue.
"""

from __future__ import annotations

import pytest

from src import config, summarize
from src.summarize import SummaryError


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Backoff is 1+2+4+8 seconds; nobody needs to wait for that in a test."""
    monkeypatch.setattr(summarize.time, "sleep", lambda _seconds: None)


@pytest.fixture(autouse=True)
def both_keys_present(monkeypatch):
    monkeypatch.setattr(config, "has_api_key", lambda provider: True)
    monkeypatch.setattr(config, "api_key", lambda provider: "test-key")


def fake_provider(*, fails: int = 0, error: str = "429 rate limit", text: str = "summary body"):
    """A provider that raises `error` the first `fails` times, then succeeds."""
    state = {"calls": 0}

    def provider(prompt: str):
        state["calls"] += 1
        if state["calls"] <= fails:
            raise RuntimeError(error)
        yield text

    provider.state = state
    return provider


# --- Transient classification --------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "429 too many requests",
        "rate limit exceeded",
        "RESOURCE_EXHAUSTED",
        "quota exceeded for this project",
        "503 UNAVAILABLE. This model is currently experiencing high demand",
        "500 internal error",
        "connection reset by peer",
        "request timed out",
    ],
)
def test_transient_failures_are_recognised(message):
    assert summarize._is_transient(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    ["400 invalid argument", "model not found", "safety block"],
)
def test_permanent_failures_are_not_retried(message):
    assert not summarize._is_transient(RuntimeError(message))


def test_rate_limit_is_distinguished_from_a_busy_provider():
    assert summarize._is_rate_limit(RuntimeError("429 rate limit"))
    assert not summarize._is_rate_limit(RuntimeError("503 unavailable"))


# --- Retry ----------------------------------------------------------------


def test_retries_a_transient_failure_then_succeeds(monkeypatch):
    provider = fake_provider(fails=2, error="503 high demand")
    monkeypatch.setitem(summarize._PROVIDERS, "gemini", provider)

    assert summarize.summarize_text("a short transcript", provider="gemini") == "summary body"
    assert provider.state["calls"] == 3


def test_gives_up_after_the_backoff_schedule(monkeypatch):
    attempts = len(config.RETRY_BACKOFF_SECONDS) + 1
    provider = fake_provider(fails=99, error="503 unavailable")
    monkeypatch.setitem(summarize._PROVIDERS, "gemini", provider)
    monkeypatch.setattr(config, "has_api_key", lambda name: name == "gemini")

    with pytest.raises(SummaryError, match="overloaded"):
        summarize.summarize_text("a short transcript", provider="gemini")
    assert provider.state["calls"] == attempts


def test_a_permanent_error_is_not_retried(monkeypatch):
    provider = fake_provider(fails=99, error="400 invalid argument")
    monkeypatch.setitem(summarize._PROVIDERS, "gemini", provider)

    with pytest.raises(SummaryError, match="returned an error"):
        summarize.summarize_text("a short transcript", provider="gemini")
    assert provider.state["calls"] == 1


# --- Provider fallback ----------------------------------------------------


def test_falls_back_to_the_other_provider_when_one_is_down(monkeypatch):
    """The whole point of splitting across two free tiers."""
    monkeypatch.setitem(summarize._PROVIDERS, "gemini", fake_provider(fails=99, error="503 down"))
    monkeypatch.setitem(summarize._PROVIDERS, "groq", fake_provider(text="from groq"))
    notices = []

    result = summarize.summarize_text(
        "a short transcript", provider="gemini", notice_callback=notices.append
    )

    assert result == "from groq"
    assert len(notices) == 1
    assert "groq" in notices[0]


def test_no_fallback_without_a_key_for_the_other_provider(monkeypatch):
    monkeypatch.setitem(summarize._PROVIDERS, "gemini", fake_provider(fails=99, error="503 down"))
    monkeypatch.setattr(config, "has_api_key", lambda name: name == "gemini")

    with pytest.raises(SummaryError):
        summarize.summarize_text("a short transcript", provider="gemini")


def test_permanent_errors_do_not_trigger_fallback(monkeypatch):
    """A malformed request fails identically on the other provider."""
    groq = fake_provider(text="from groq")
    monkeypatch.setitem(summarize._PROVIDERS, "gemini", fake_provider(fails=99, error="400 bad"))
    monkeypatch.setitem(summarize._PROVIDERS, "groq", groq)

    with pytest.raises(SummaryError):
        summarize.summarize_text("a short transcript", provider="gemini")
    assert groq.state["calls"] == 0


def test_the_fallback_notice_fires_only_once(monkeypatch):
    """Map-reduce makes many calls; the user needs telling once, not nine times."""
    monkeypatch.setattr(config, "SUMMARY_INPUT_LIMIT_CHARS", 100)
    monkeypatch.setattr(config, "MAP_CHUNK_CHARS", 60)
    monkeypatch.setitem(summarize._PROVIDERS, "gemini", fake_provider(fails=99, error="503 down"))
    monkeypatch.setitem(summarize._PROVIDERS, "groq", fake_provider(text="part"))
    notices = []

    summarize.summarize_text("x " * 200, provider="gemini", notice_callback=notices.append)

    assert len(notices) == 1


# --- Map-reduce splitting -------------------------------------------------


def test_short_text_is_a_single_part():
    assert len(summarize.split_for_map("short enough")) == 1


def test_long_text_splits_into_several_parts(monkeypatch):
    monkeypatch.setattr(config, "MAP_CHUNK_CHARS", 500)
    parts = summarize.split_for_map("word " * 1000)

    assert len(parts) > 1
    assert all(part.strip() for part in parts)


def test_split_covers_the_whole_transcript(monkeypatch):
    """Silently dropping the tail is the failure this guards against."""
    monkeypatch.setattr(config, "MAP_CHUNK_CHARS", 400)
    text = " ".join(f"sentence{n}." for n in range(300))
    parts = summarize.split_for_map(text)

    assert parts[0].startswith("sentence0.")
    assert text.endswith(parts[-1].strip()[-20:])


def test_split_prefers_sentence_boundaries(monkeypatch):
    monkeypatch.setattr(config, "MAP_CHUNK_CHARS", 200)
    monkeypatch.setattr(config, "MAP_CHUNK_OVERLAP_CHARS", 0)
    text = ("This is a full sentence of a reasonable length. " * 20).strip()
    parts = summarize.split_for_map(text)

    # Every part but the last should end where a sentence ended.
    assert all(part.strip().endswith(".") for part in parts[:-1])


def test_split_handles_the_bangla_full_stop(monkeypatch):
    """'।' ends a Bangla sentence; the English boundaries miss it entirely."""
    monkeypatch.setattr(config, "MAP_CHUNK_CHARS", 200)
    monkeypatch.setattr(config, "MAP_CHUNK_OVERLAP_CHARS", 0)
    text = ("আমরা আজ একটি সভা করেছি। " * 40).strip()
    parts = summarize.split_for_map(text)

    assert len(parts) > 1
    assert all(part.strip().endswith("।") for part in parts[:-1])


def test_map_reduce_runs_every_part_plus_a_merge(monkeypatch):
    monkeypatch.setattr(config, "SUMMARY_INPUT_LIMIT_CHARS", 100)
    monkeypatch.setattr(config, "MAP_CHUNK_CHARS", 60)
    provider = fake_provider(text="chunk summary")
    monkeypatch.setitem(summarize._PROVIDERS, "gemini", provider)
    steps = []

    summarize.summarize_text(
        "x " * 200, provider="gemini", progress_callback=lambda c, t: steps.append((c, t))
    )

    parts = len(summarize.split_for_map("x " * 200))
    assert provider.state["calls"] == parts + 1  # one per part, plus the merge
    assert steps[-1] == (parts + 1, parts + 1)


# --- Guards ---------------------------------------------------------------


def test_empty_transcript_raises():
    with pytest.raises(SummaryError, match="no transcript"):
        summarize.summarize_text("   ")


def test_unknown_provider_raises():
    with pytest.raises(SummaryError, match="Unknown LLM provider"):
        summarize.summarize_text("something", provider="nonsense")
