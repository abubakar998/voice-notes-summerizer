"""Smoke tests for the Streamlit layer.

`AppTest` executes app.py the way the server would, so a typo or a bad API call
in the UI fails here instead of in front of a user. No browser, no network:
the results path is driven entirely from pre-seeded session state.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from .conftest import speech_like, to_wav_bytes

# AppTest resolves a relative path against the calling file, which would look
# for tests/app.py.
APP_PATH = Path(__file__).resolve().parent.parent / "app.py"

TRANSCRIPT = [
    {"start": 0.0, "end": 4.0, "text": "Welcome to the planning call."},
    {"start": 4.0, "end": 9.0, "text": "Rafi will send the draft on Friday."},
]

SUMMARY = "## Summary\nA planning call.\n\n## Action Items\n- Rafi: send the draft."


def app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=60)


def seeded_app() -> AppTest:
    """An app that already holds a finished result, as after a run."""
    audio_bytes = to_wav_bytes(speech_like(6_000))
    digest = hashlib.sha256(audio_bytes).hexdigest()

    at = app()
    at.session_state["audio_bytes"] = audio_bytes
    at.session_state["filename"] = "standup.wav"
    at.session_state["audio_hash"] = digest
    at.session_state["processed_hash"] = digest
    at.session_state["transcript"] = TRANSCRIPT
    at.session_state["summary"] = SUMMARY
    at.session_state["metadata"] = {
        "filename": "standup.wav",
        "duration": 6.0,
        "language": "en",
        "word_count": 12,
        "processing_seconds": 3.2,
        "generated_at": "2026-08-17 00:00",
        "asr_model": "whisper-large-v3-turbo",
        "llm_model": "gemini-flash-latest",
        "notices": [],
    }
    return at


# --- Empty state ----------------------------------------------------------


def test_app_runs_without_input():
    at = app().run()

    assert not at.exception


def test_empty_state_prompts_for_a_recording():
    at = app().run()

    assert any("Upload a file" in info.value for info in at.info)


def test_sidebar_offers_upload_and_recording():
    at = app().run()

    assert len(at.sidebar.file_uploader) == 1
    assert len(at.sidebar.selectbox) == 1  # language override


def test_language_options_include_auto_detect_and_bangla():
    at = app().run()
    options = at.sidebar.selectbox[0].options

    assert "Auto-detect" in options
    assert "Bangla" in options


# --- Results state --------------------------------------------------------


def test_results_render_without_error():
    at = seeded_app().run()

    assert not at.exception


def test_summary_is_shown():
    at = seeded_app().run()
    body = " ".join(element.value for element in at.markdown)

    assert "A planning call." in body


def test_metadata_row_is_populated():
    at = seeded_app().run()
    labels = {metric.label for metric in at.metric}

    assert {"Duration", "Language", "Words", "Processed in"} <= labels


def test_both_downloads_are_offered():
    at = seeded_app().run()
    names = [button.label for button in at.download_button]

    assert "Download .md" in names
    assert "Download .txt" in names


def test_transcript_is_collapsed_behind_an_expander():
    """The summary is what people came for; the transcript is reference."""
    at = seeded_app().run()

    assert any("transcript" in expander.label.lower() for expander in at.expander)


def test_reprocessing_the_same_audio_is_blocked():
    """Hashing the audio is what stops a rerun retriggering a 60-second job."""
    at = seeded_app().run()
    run_buttons = [b for b in at.button if "Transcribe" in b.label]

    assert run_buttons and run_buttons[0].disabled


def test_start_over_clears_the_results():
    at = seeded_app().run()
    start_over = next(b for b in at.sidebar.button if "Start over" in b.label)

    start_over.click().run()

    assert at.session_state["audio_bytes"] is None
    assert at.session_state["transcript"] is None
    assert not at.exception


def test_a_fallback_notice_is_surfaced_to_the_user():
    at = seeded_app()
    at.session_state["metadata"]["notices"] = ["gemini was unavailable, used groq"]
    at.run()

    assert any("groq" in info.value for info in at.info)


# --- Failure surfaces -----------------------------------------------------


def test_undecodable_audio_shows_a_readable_error():
    at = app()
    audio_bytes = b"definitely not audio"
    digest = hashlib.sha256(audio_bytes).hexdigest()
    at.session_state["audio_bytes"] = audio_bytes
    at.session_state["filename"] = "broken.wav"
    at.session_state["audio_hash"] = digest
    at.run()

    assert at.error
    assert "Could not decode" in at.error[0].value


@pytest.mark.parametrize("summary", ["", None])
def test_a_missing_summary_still_renders_the_transcript(summary):
    """Transcription is the expensive half — a failed summary must not lose it."""
    at = seeded_app()
    at.session_state["summary"] = summary
    at.run()

    assert not at.exception
    assert at.download_button
