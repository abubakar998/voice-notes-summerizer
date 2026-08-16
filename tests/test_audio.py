"""Normalization, splitting and input validation."""

from __future__ import annotations

import pytest
from pydub import AudioSegment

from src import config
from src.audio import AudioError, get_duration_seconds, load_and_normalize, split_audio

from .conftest import speech_like, to_wav_bytes


# --- Normalization --------------------------------------------------------


def test_normalizes_to_16khz_mono():
    stereo = speech_like(5_000).set_frame_rate(44_100).set_channels(2)
    audio = load_and_normalize(to_wav_bytes(stereo), "note.wav")

    assert audio.frame_rate == config.TARGET_SAMPLE_RATE
    assert audio.channels == config.TARGET_CHANNELS
    assert audio.sample_width == config.TARGET_SAMPLE_WIDTH


def test_normalization_preserves_duration():
    original = speech_like(8_000).set_frame_rate(44_100).set_channels(2)
    audio = load_and_normalize(to_wav_bytes(original), "note.wav")

    assert get_duration_seconds(audio) == pytest.approx(8.0, abs=0.1)


def test_normalization_shrinks_the_file():
    """The size win is the reason this happens before anything else."""
    stereo = speech_like(10_000).set_frame_rate(44_100).set_channels(2)
    raw = to_wav_bytes(stereo)
    normalized = to_wav_bytes(load_and_normalize(raw, "note.wav"))

    assert len(normalized) < len(raw) / 4


# --- Input validation -----------------------------------------------------


def test_silent_file_raises():
    silence = AudioSegment.silent(duration=5_000, frame_rate=16_000)
    with pytest.raises(AudioError, match="silent"):
        load_and_normalize(to_wav_bytes(silence), "quiet.wav")


def test_clip_under_one_second_raises():
    with pytest.raises(AudioError, match="under a second|only"):
        load_and_normalize(to_wav_bytes(speech_like(400, tone_ms=400, gap_ms=0)), "tiny.wav")


def test_empty_file_raises():
    with pytest.raises(AudioError, match="empty"):
        load_and_normalize(b"", "nothing.wav")


def test_oversized_file_is_rejected_before_decoding():
    oversized = b"\x00" * int((config.MAX_UPLOAD_MB + 1) * 1024 * 1024)
    with pytest.raises(AudioError, match="limit"):
        load_and_normalize(oversized, "huge.wav")


def test_undecodable_file_raises_clearly():
    with pytest.raises(AudioError, match="Could not decode"):
        load_and_normalize(b"this is not audio at all", "broken.wav")


# --- Splitting ------------------------------------------------------------


def test_short_audio_is_one_chunk_at_offset_zero(short_clip):
    chunks = split_audio(short_clip)

    assert len(chunks) == 1
    assert chunks[0]["start_ms"] == 0


def test_long_audio_splits_into_expected_chunk_count(long_clip):
    """25 minutes at a 10-minute target is three chunks."""
    assert len(split_audio(long_clip)) == 3


def test_chunk_offsets_are_ordered_and_start_at_zero(long_clip):
    chunks = split_audio(long_clip)
    offsets = [chunk["start_ms"] for chunk in chunks]

    assert offsets[0] == 0
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == len(offsets)


def test_chunk_offsets_land_near_the_target_boundaries(long_clip):
    """Offsets may shift to reach a silence, but only within the search window."""
    chunks = split_audio(long_clip)

    for index, chunk in enumerate(chunks):
        expected = index * config.CHUNK_LENGTH_MS
        drift = abs(chunk["start_ms"] - expected)
        assert drift <= config.SILENCE_SEARCH_WINDOW_MS


def test_chunks_cover_the_whole_recording(long_clip):
    """Every millisecond must appear in some chunk, or speech goes missing."""
    chunks = split_audio(long_clip)

    assert chunks[0]["start_ms"] == 0
    for index in range(len(chunks) - 1):
        chunk_end = chunks[index]["start_ms"] + len(chunks[index]["audio"])
        assert chunk_end >= chunks[index + 1]["start_ms"]

    last = chunks[-1]
    assert last["start_ms"] + len(last["audio"]) == len(long_clip)


def test_chunks_overlap_their_successor(long_clip):
    """Overlap is what stops a word on a seam being lost from both sides."""
    chunks = split_audio(long_clip)

    for index in range(len(chunks) - 1):
        chunk_end = chunks[index]["start_ms"] + len(chunks[index]["audio"])
        overlap = chunk_end - chunks[index + 1]["start_ms"]
        assert overlap == pytest.approx(config.CHUNK_OVERLAP_MS, abs=50)


def test_splits_land_in_silence(long_clip):
    """A cut through a tone burst would garble a real word."""
    chunks = split_audio(long_clip)

    for chunk in chunks[1:]:
        boundary = chunk["start_ms"]
        at_cut = long_clip[boundary - 100 : boundary + 100]
        assert at_cut.dBFS < -40, f"cut at {boundary}ms landed in speech"


def test_split_falls_back_to_a_hard_cut_on_continuous_speech():
    """No silence anywhere still has to produce usable chunks."""
    continuous = speech_like(22 * 60 * 1000, tone_ms=60_000, gap_ms=0)
    chunks = split_audio(continuous)

    assert len(chunks) >= 2
    assert chunks[0]["start_ms"] == 0
    last = chunks[-1]
    assert last["start_ms"] + len(last["audio"]) == len(continuous)
