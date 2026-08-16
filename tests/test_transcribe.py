"""Stitching chunks back together.

Offset correction and overlap de-duplication are the two places this pipeline
is most likely to be subtly wrong, and both are invisible on a short file.
"""

from __future__ import annotations

import pytest
from pydub import AudioSegment

from src import config, transcribe
from src.transcribe import (
    TranscriptionError,
    _drop_leading_words,
    _overlap_word_count,
    stitch,
)


def segment(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text}


# --- Offsets --------------------------------------------------------------


def test_first_chunk_timestamps_are_unchanged():
    result = stitch([{"segments": [segment(0.0, 4.0, "hello there")], "start_ms": 0}])

    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 4.0


def test_later_chunk_timestamps_get_their_offset_added():
    """Whisper reports a chunk starting at 10:00 as beginning at 0:00."""
    result = stitch(
        [
            {"segments": [segment(0.0, 5.0, "first chunk")], "start_ms": 0},
            {"segments": [segment(2.0, 7.0, "second chunk")], "start_ms": 600_000},
        ]
    )

    assert result[1]["start"] == 602.0
    assert result[1]["end"] == 607.0


def test_final_timestamp_matches_the_recording_length():
    """The invariant that catches broken offsets on a long file."""
    chunks = [
        {"segments": [segment(0.0, 595.0, "part one")], "start_ms": 0},
        {"segments": [segment(0.0, 595.0, "part two")], "start_ms": 600_000},
        {"segments": [segment(0.0, 590.0, "part three")], "start_ms": 1_200_000},
    ]
    result = stitch(chunks)

    assert result[-1]["end"] == pytest.approx(1790.0)


def test_timestamps_never_go_backwards():
    result = stitch(
        [
            {"segments": [segment(0.0, 9.0, "alpha"), segment(9.0, 18.0, "beta")], "start_ms": 0},
            {"segments": [segment(0.0, 9.0, "gamma")], "start_ms": 20_000},
        ]
    )
    starts = [item["start"] for item in result]

    assert starts == sorted(starts)


# --- Overlap de-duplication ----------------------------------------------


def test_detects_a_repeated_phrase_across_a_seam():
    previous = [segment(0.0, 6.0, "we should ship the draft on Friday")]
    following = [segment(0.0, 6.0, "the draft on Friday and then review it")]

    assert _overlap_word_count(previous, following) == 4


def test_detects_no_overlap_when_there_is_none():
    previous = [segment(0.0, 6.0, "we should ship the draft")]
    following = [segment(0.0, 6.0, "completely unrelated sentence here")]

    assert _overlap_word_count(previous, following) == 0


def test_overlap_ignores_punctuation_and_case():
    previous = [segment(0.0, 6.0, "that is the plan, Rafi.")]
    following = [segment(0.0, 6.0, "The plan Rafi — we start Monday")]

    assert _overlap_word_count(previous, following) == 3


def test_overlap_prefers_the_longest_match():
    """A one-word coincidence must not win over the real seam."""
    previous = [segment(0.0, 6.0, "the budget is the main issue")]
    following = [segment(0.0, 6.0, "the main issue is the timeline")]

    assert _overlap_word_count(previous, following) == 3


def test_stitch_removes_the_duplicated_words():
    result = stitch(
        [
            {"segments": [segment(0.0, 6.0, "we ship the draft on Friday")], "start_ms": 0},
            {
                "segments": [segment(0.0, 6.0, "the draft on Friday and then we review")],
                "start_ms": 5_000,
            },
        ]
    )
    joined = " ".join(item["text"] for item in result)

    assert joined == "we ship the draft on Friday and then we review"


def test_stitch_leaves_distinct_chunks_alone():
    result = stitch(
        [
            {"segments": [segment(0.0, 6.0, "first topic entirely")], "start_ms": 0},
            {"segments": [segment(0.0, 6.0, "second topic entirely")], "start_ms": 6_000},
        ]
    )
    joined = " ".join(item["text"] for item in result)

    assert joined == "first topic entirely second topic entirely"


def test_drop_leading_words_trims_within_a_segment():
    segments = [segment(0.0, 3.0, "one two three four")]

    assert _drop_leading_words(segments, 2)[0]["text"] == "three four"


def test_drop_leading_words_discards_fully_consumed_segments():
    segments = [segment(0.0, 3.0, "one two"), segment(3.0, 6.0, "three four")]
    result = _drop_leading_words(segments, 3)

    assert len(result) == 1
    assert result[0]["text"] == "four"


def test_drop_leading_words_can_empty_the_list():
    segments = [segment(0.0, 3.0, "one two")]

    assert _drop_leading_words(segments, 5) == []


def test_drop_leading_words_ignores_standalone_punctuation():
    """The overlap is counted in normalized words, so a lone dash is not one.

    Counting it would shift the trim by a token and either duplicate or eat a
    real word at the seam.
    """
    segments = [segment(0.0, 3.0, "one — two three")]

    assert _drop_leading_words(segments, 2)[0]["text"] == "three"


def test_overlap_trim_survives_punctuation_between_words():
    result = stitch(
        [
            {"segments": [segment(0.0, 6.0, "we ship the draft on Friday")], "start_ms": 0},
            {
                "segments": [segment(0.0, 6.0, "the draft on Friday — and then we review")],
                "start_ms": 5_000,
            },
        ]
    )

    assert " ".join(i["text"] for i in result) == "we ship the draft on Friday — and then we review"


def test_duplicate_chunk_collapses_entirely():
    """Worst case: a chunk that repeats its predecessor word for word."""
    text = "exactly the same words twice over"
    result = stitch(
        [
            {"segments": [segment(0.0, 6.0, text)], "start_ms": 0},
            {"segments": [segment(0.0, 6.0, text)], "start_ms": 3_000},
        ]
    )

    assert " ".join(item["text"] for item in result) == text


# --- Language detection ---------------------------------------------------


def test_dominant_language_is_a_majority_vote():
    """Per-chunk detection disagrees on code-switched speech."""
    assert transcribe._dominant_language(["bn", "en", "bn"]) == "bn"


def test_dominant_language_of_nothing_is_none():
    assert transcribe._dominant_language([]) is None


# --- Failure handling -----------------------------------------------------


def _one_chunk(duration_ms: int = 5_000, start_ms: int = 0) -> dict:
    return {"audio": AudioSegment.silent(duration=duration_ms, frame_rate=16_000), "start_ms": start_ms}


def test_a_failed_chunk_is_marked_inline_and_the_rest_survive(monkeypatch):
    calls = {"n": 0}

    def flaky(chunk_bytes, language):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("provider exploded")
        return {
            "text": "second chunk fine",
            "segments": [{"start": 0.0, "end": 4.0, "text": "second chunk fine"}],
            "language": "en",
            "duration": 4.0,
        }

    monkeypatch.setitem(transcribe._PROVIDERS, "local", flaky)
    result = transcribe.transcribe(
        [_one_chunk(start_ms=0), _one_chunk(start_ms=600_000)], provider="local"
    )

    assert result["failed_chunks"] == 1
    assert "transcription failed for 00:00" in result["text"]
    assert "second chunk fine" in result["text"]


def test_every_chunk_failing_raises():
    def always_fails(chunk_bytes, language):
        raise RuntimeError("provider down")

    original = transcribe._PROVIDERS["local"]
    transcribe._PROVIDERS["local"] = always_fails
    try:
        with pytest.raises(TranscriptionError, match="Every chunk failed"):
            transcribe.transcribe([_one_chunk()], provider="local")
    finally:
        transcribe._PROVIDERS["local"] = original


def test_a_missing_local_model_reports_how_to_install_it(monkeypatch):
    """Setup problems fail on every chunk, so the fixable message must survive.

    Without the preflight this surfaced as "every chunk failed — or set
    ASR_PROVIDER=local", advice the user had already followed.
    """

    def missing():
        raise TranscriptionError("faster-whisper is not installed. Run pip install ...")

    monkeypatch.setattr(transcribe, "_load_local_model", missing)

    with pytest.raises(TranscriptionError, match="faster-whisper is not installed"):
        transcribe.transcribe([_one_chunk()], provider="local")


def test_a_missing_groq_key_is_reported_before_any_upload(monkeypatch):
    def no_key(provider):
        raise RuntimeError("GROQ_API_KEY is not set. Get a free key at ...")

    monkeypatch.setattr(config, "api_key", no_key)

    with pytest.raises(TranscriptionError, match="GROQ_API_KEY"):
        transcribe.transcribe([_one_chunk()], provider="groq")


def test_unknown_provider_raises():
    with pytest.raises(TranscriptionError, match="Unknown ASR_PROVIDER"):
        transcribe.transcribe([_one_chunk()], provider="nonsense")


def test_no_audio_raises():
    with pytest.raises(TranscriptionError, match="no audio"):
        transcribe.transcribe([], provider="local")


def test_progress_callback_fires_once_per_chunk(monkeypatch):
    def ok(chunk_bytes, language):
        return {"text": "x", "segments": [], "language": "en", "duration": 1.0}

    monkeypatch.setitem(transcribe._PROVIDERS, "local", ok)
    seen = []
    transcribe.transcribe(
        [_one_chunk(start_ms=0), _one_chunk(start_ms=600_000)],
        progress_callback=lambda current, total: seen.append((current, total)),
        provider="local",
    )

    assert seen == [(1, 2), (2, 2)]
