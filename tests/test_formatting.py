"""Timestamp rendering and export structure."""

from __future__ import annotations

from src.formatting import (
    count_words,
    format_duration,
    format_timestamp,
    format_transcript,
    language_name,
    to_markdown,
    to_plaintext,
)

TRANSCRIPT = [
    {"start": 0.0, "end": 4.2, "text": "Welcome to the planning call."},
    {"start": 4.2, "end": 9.8, "text": "Rafi will send the draft on Friday."},
    {"start": 3661.0, "end": 3665.0, "text": "That is everything for today."},
]

SUMMARY = """## Summary
A short planning call.

## Key Points
- The draft is due Friday.

## Action Items
- Rafi: send the draft by Friday."""

METADATA = {
    "filename": "standup.m4a",
    "duration": 3665.0,
    "language": "bn",
    "word_count": 18,
    "processing_seconds": 12.4,
    "generated_at": "2026-08-16 21:45",
    "asr_model": "whisper-large-v3-turbo",
    "llm_model": "gemini-flash-latest",
}


# --- Timestamps -----------------------------------------------------------


def test_formats_under_a_minute():
    assert format_timestamp(7) == "00:07"


def test_formats_minutes_and_seconds():
    assert format_timestamp(125) == "02:05"


def test_crosses_the_hour_boundary():
    """59:59 stays MM:SS; one second later it has to grow an hours field."""
    assert format_timestamp(3599) == "59:59"
    assert format_timestamp(3600) == "1:00:00"
    assert format_timestamp(3661) == "1:01:01"


def test_rounds_fractional_seconds():
    assert format_timestamp(12.6) == "00:13"


def test_negative_and_none_are_treated_as_zero():
    assert format_timestamp(-5) == "00:00"
    assert format_timestamp(None) == "00:00"


def test_format_duration_reads_naturally():
    assert format_duration(65) == "1m 05s"
    assert format_duration(3665) == "1h 01m 05s"


# --- Small helpers --------------------------------------------------------


def test_language_name_maps_known_codes():
    assert language_name("bn") == "Bangla"
    assert language_name("en") == "English"


def test_language_name_passes_through_unknown_codes():
    assert language_name("xx") == "xx"
    assert language_name(None) == "Unknown"


def test_count_words():
    assert count_words("one two three") == 3
    assert count_words("") == 0


def test_transcript_lines_are_timestamped():
    lines = format_transcript(TRANSCRIPT).splitlines()

    assert lines[0] == "[00:00] Welcome to the planning call."
    assert lines[2].startswith("[1:01:01]")


def test_transcript_skips_empty_segments():
    segments = [{"start": 0.0, "end": 1.0, "text": "   "}] + TRANSCRIPT
    assert len(format_transcript(segments).splitlines()) == len(TRANSCRIPT)


# --- Markdown export ------------------------------------------------------


def test_markdown_contains_every_section():
    output = to_markdown(TRANSCRIPT, SUMMARY, METADATA)

    for expected in ("## Summary", "## Key Points", "## Action Items", "## Full Transcript"):
        assert expected in output


def test_markdown_header_carries_the_metadata():
    output = to_markdown(TRANSCRIPT, SUMMARY, METADATA)

    assert "standup.m4a" in output
    assert "1h 01m 05s" in output
    assert "Bangla" in output
    assert "2026-08-16 21:45" in output


def test_markdown_puts_the_summary_before_the_transcript():
    """People want the summary first; the transcript is reference material."""
    output = to_markdown(TRANSCRIPT, SUMMARY, METADATA)

    assert output.index("## Summary") < output.index("## Full Transcript")


def test_markdown_includes_the_timestamped_transcript():
    output = to_markdown(TRANSCRIPT, SUMMARY, METADATA)

    assert "[00:00] Welcome to the planning call." in output
    assert "[1:01:01] That is everything for today." in output


def test_markdown_handles_a_missing_summary():
    output = to_markdown(TRANSCRIPT, "", METADATA)

    assert "No summary generated" in output
    assert "## Full Transcript" in output


# --- Plain text export ----------------------------------------------------


def test_plaintext_has_no_markdown_syntax():
    output = to_plaintext(TRANSCRIPT, SUMMARY, METADATA)

    assert "##" not in output
    assert "**" not in output


def test_plaintext_keeps_the_same_content():
    output = to_plaintext(TRANSCRIPT, SUMMARY, METADATA)

    assert "Summary" in output
    assert "Action Items" in output
    assert "FULL TRANSCRIPT" in output
    assert "Rafi" in output
    assert "standup.m4a" in output


def test_plaintext_strips_bold_but_keeps_the_words():
    output = to_plaintext(TRANSCRIPT, "**Important** decision made.", METADATA)

    assert "Important decision made." in output
