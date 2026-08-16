"""Timestamp formatting and export rendering."""

from __future__ import annotations

import re
from datetime import datetime

# What Whisper reports, mapped to something a human recognises. The two
# providers disagree on format — Groq's verbose_json returns a full name
# ("English"), faster-whisper returns an ISO code ("en") — so both are keys
# here. "Bengali" is the ISO name; "Bangla" is what its speakers call it.
LANGUAGE_NAMES = {
    "bn": "Bangla",
    "bengali": "Bangla",
    "en": "English",
    "english": "English",
    "hi": "Hindi",
    "hindi": "Hindi",
    "ur": "Urdu",
    "urdu": "Urdu",
    "ar": "Arabic",
    "arabic": "Arabic",
    "es": "Spanish",
    "spanish": "Spanish",
    "fr": "French",
    "french": "French",
    "de": "German",
    "german": "German",
}


def format_timestamp(seconds: float) -> str:
    """Render seconds as MM:SS, or HH:MM:SS once past an hour."""
    if seconds is None or seconds < 0:
        seconds = 0
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """Human-readable length, e.g. '12m 04s'."""
    total = int(round(seconds or 0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def language_name(code: str | None) -> str:
    """Whatever the provider reported, rendered for a human."""
    if not code:
        return "Unknown"
    known = LANGUAGE_NAMES.get(code.lower())
    if known:
        return known
    # An unrecognised full name still reads better capitalized than an ISO code
    # does expanded, so only touch the ones that are clearly words.
    return code.capitalize() if len(code) > 3 else code


def count_words(text: str) -> int:
    return len(text.split()) if text else 0


def format_transcript(segments: list[dict]) -> str:
    """One timestamped line per segment."""
    lines = []
    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{format_timestamp(segment.get('start', 0))}] {text}")
    return "\n".join(lines)


def _header_lines(metadata: dict) -> list[str]:
    generated = metadata.get("generated_at") or datetime.now().strftime(
        "%Y-%m-%d %H:%M"
    )
    lines = [
        f"File: {metadata.get('filename', 'recording')}",
        f"Duration: {format_duration(metadata.get('duration', 0))}",
        f"Detected language: {language_name(metadata.get('language'))}",
        f"Words: {metadata.get('word_count', 0)}",
        f"Generated: {generated}",
    ]
    if metadata.get("processing_seconds"):
        lines.append(f"Processing time: {metadata['processing_seconds']:.1f}s")
    if metadata.get("asr_model"):
        lines.append(f"Transcribed with: {metadata['asr_model']}")
    if metadata.get("llm_model"):
        lines.append(f"Summarized with: {metadata['llm_model']}")
    return lines


def to_markdown(transcript: list[dict], summary: str, metadata: dict) -> str:
    """Summary first, then the timestamped transcript. Summary is what people read."""
    parts = [f"# Voice Note — {metadata.get('filename', 'recording')}", ""]
    parts.extend(f"- {line}" for line in _header_lines(metadata))
    parts.extend(["", "---", "", (summary or "_No summary generated._").strip(), ""])
    parts.extend(["---", "", "## Full Transcript", ""])

    body = format_transcript(transcript)
    parts.append(body if body else "_No speech was transcribed._")
    parts.append("")
    return "\n".join(parts)


def to_plaintext(transcript: list[dict], summary: str, metadata: dict) -> str:
    """Same content as the markdown export, with the syntax stripped out."""
    title = f"VOICE NOTE — {metadata.get('filename', 'recording')}"
    parts = [title, "=" * len(title), ""]
    parts.extend(_header_lines(metadata))
    parts.extend(["", "-" * 60, ""])

    clean = _strip_markdown(summary or "No summary generated.")
    parts.extend([clean.strip(), "", "-" * 60, "", "FULL TRANSCRIPT", ""])

    body = format_transcript(transcript)
    parts.append(body if body else "No speech was transcribed.")
    parts.append("")
    return "\n".join(parts)


def _strip_markdown(text: str) -> str:
    """Remove the markdown the model emits, keeping the structure readable."""
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^#{1,6}\s*", "", line)  # headings
        line = re.sub(r"^\s*[-*+]\s+", "  - ", line)  # bullets
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)  # bold
        line = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"\1", line)  # italic
        line = re.sub(r"`(.+?)`", r"\1", line)  # inline code
        lines.append(line.rstrip())
    return "\n".join(lines)
