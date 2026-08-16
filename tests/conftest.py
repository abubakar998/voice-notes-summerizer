"""Shared fixtures. All test audio is generated, never committed as files."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def speech_like(duration_ms: int, tone_ms: int = 9000, gap_ms: int = 1000) -> AudioSegment:
    """Tone bursts separated by silence — a stand-in for speech with pauses.

    The gaps are what the splitter is supposed to find and cut on, so their
    spacing is deliberate: one every `tone_ms + gap_ms`.
    """
    cycle = Sine(440).to_audio_segment(duration=tone_ms).apply_gain(-6) + AudioSegment.silent(
        duration=gap_ms
    )
    repeats = duration_ms // len(cycle) + 1
    return (cycle * repeats)[:duration_ms].set_frame_rate(16000).set_channels(1)


def to_wav_bytes(audio: AudioSegment) -> bytes:
    """Encode as WAV, which pydub writes without needing ffmpeg."""
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def short_clip() -> AudioSegment:
    return speech_like(20_000)


@pytest.fixture(scope="session")
def long_clip() -> AudioSegment:
    """25 minutes — long enough to force three chunks at a 10-minute target."""
    return speech_like(25 * 60 * 1000)
