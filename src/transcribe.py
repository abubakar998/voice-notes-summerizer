"""Speech-to-text, with Groq and faster-whisper behind one function.

Groq serves Whisper on custom inference hardware and comes back many times
faster than real time, which is what makes this usable on a free CPU container.
faster-whisper is the offline fallback: slower, but needs no network and no key,
and having both is what makes the §11 benchmark possible.

The subtle part is joining chunks back together. Whisper timestamps every clip
from zero, so each chunk's offset has to be added back, and because chunks
deliberately overlap, the words on each seam arrive twice and have to be
trimmed.
"""

from __future__ import annotations

import re
import time
from typing import Callable

from . import audio as audio_module
from . import config
from .formatting import format_timestamp


class TranscriptionError(RuntimeError):
    """A provider failure, already phrased for a human to read."""


RATE_LIMIT_MESSAGE = (
    "Groq is rate limiting us (free tier). Wait a minute and try again, or set "
    "ASR_PROVIDER=local in .env to transcribe on this machine instead."
)

BUSY_MESSAGE = (
    "Groq is temporarily overloaded and did not recover after several retries. "
    "Wait a minute and try again, or set ASR_PROVIDER=local in .env."
)

_LOCAL_MISSING = (
    "ASR_PROVIDER is 'local' but faster-whisper is not installed. Run "
    "`pip install -r requirements-dev.txt`, or set ASR_PROVIDER=groq in .env."
)

# Cached across calls — loading a faster-whisper model takes seconds and the
# Streamlit app would otherwise pay that on every single run.
_local_model = None
_local_model_size: str | None = None


def _is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "429" in text or "rate limit" in text or "quota" in text


def _is_transient(exc: Exception) -> bool:
    """Worth retrying: throttling, or the provider briefly falling over.

    Free tiers serve 503 "high demand" as routinely as they serve 429, and a
    60-minute recording is six upload requests — enough to catch one.
    """
    if _is_rate_limit(exc):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return (
        "503" in text
        or "500" in text
        or "502" in text
        or "504" in text
        or "unavailable" in text
        or "overloaded" in text
        or "high demand" in text
        or "internal error" in text
        or "timeout" in text
        or "timed out" in text
        or "connection" in text
    )


def _normalize_words(text: str) -> list[str]:
    """Lowercase, drop punctuation, split. Used only for comparing overlaps."""
    cleaned = re.sub(r"[^\w\sঀ-৿]", " ", text.lower())
    return cleaned.split()


# --- Groq -----------------------------------------------------------------


def _transcribe_groq(chunk_bytes: bytes, language: str | None) -> dict:
    from groq import Groq

    client = Groq(api_key=config.api_key("groq"))

    size_mb = len(chunk_bytes) / (1024 * 1024)
    if size_mb > config.GROQ_MAX_FILE_MB:
        raise TranscriptionError(
            f"A {size_mb:.0f}MB chunk exceeds Groq's {config.GROQ_MAX_FILE_MB}MB "
            "limit. Lower CHUNK_LENGTH_MS in src/config.py."
        )

    kwargs = {
        "file": (f"chunk.{config.CHUNK_UPLOAD_FORMAT}", chunk_bytes),
        "model": config.GROQ_ASR_MODEL,
        "response_format": "verbose_json",  # required for segment timestamps
        "timestamp_granularities": ["segment"],
    }
    if language:
        kwargs["language"] = language

    response = client.audio.transcriptions.create(**kwargs)
    return _parse_groq_response(response)


def _parse_groq_response(response) -> dict:
    """Normalize the SDK's response, which may be a model or a plain dict."""

    def field(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    raw_segments = field(response, "segments") or []
    segments = []
    for item in raw_segments:
        text = (field(item, "text") or "").strip()
        if not text:
            continue
        segments.append(
            {
                "start": float(field(item, "start", 0.0) or 0.0),
                "end": float(field(item, "end", 0.0) or 0.0),
                "text": text,
            }
        )

    text = (field(response, "text") or "").strip()
    if not text and segments:
        text = " ".join(segment["text"] for segment in segments)

    return {
        "text": text,
        "segments": segments,
        "language": field(response, "language"),
        "duration": float(field(response, "duration", 0.0) or 0.0),
    }


# --- faster-whisper (local) -----------------------------------------------


def _load_local_model():
    global _local_model, _local_model_size
    if _local_model is not None and _local_model_size == config.LOCAL_ASR_MODEL:
        return _local_model
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(_LOCAL_MISSING) from exc

    # int8 on CPU is the only combination that is tolerable without a GPU.
    _local_model = WhisperModel(config.LOCAL_ASR_MODEL, device="cpu", compute_type="int8")
    _local_model_size = config.LOCAL_ASR_MODEL
    return _local_model


def _transcribe_local(chunk_bytes: bytes, language: str | None) -> dict:
    import io

    model = _load_local_model()
    segments_iter, info = model.transcribe(
        io.BytesIO(chunk_bytes),
        language=language,
        beam_size=5,
        vad_filter=True,  # skips silence, which Whisper otherwise hallucinates over
    )

    segments = []
    for segment in segments_iter:  # lazy generator; this is where the work happens
        text = (segment.text or "").strip()
        if text:
            segments.append(
                {"start": float(segment.start), "end": float(segment.end), "text": text}
            )

    return {
        "text": " ".join(segment["text"] for segment in segments),
        "segments": segments,
        "language": getattr(info, "language", None),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
    }


_PROVIDERS = {"groq": _transcribe_groq, "local": _transcribe_local}


def _transcribe_chunk_with_retry(
    chunk_bytes: bytes, language: str | None, provider: str
) -> dict:
    """One chunk, retrying 429s with exponential backoff before giving up."""
    transcribe_fn = _PROVIDERS[provider]
    for delay in [*config.RETRY_BACKOFF_SECONDS, None]:
        try:
            return transcribe_fn(chunk_bytes, language)
        except TranscriptionError:
            raise
        except Exception as exc:
            if _is_transient(exc) and delay is not None:
                time.sleep(delay)
                continue
            if _is_transient(exc):
                raise TranscriptionError(
                    RATE_LIMIT_MESSAGE if _is_rate_limit(exc) else BUSY_MESSAGE
                ) from exc
            raise TranscriptionError(f"Transcription failed: {exc}") from exc
    raise TranscriptionError(RATE_LIMIT_MESSAGE)


# --- Stitching ------------------------------------------------------------


def _overlap_word_count(previous: list[dict], following: list[dict]) -> int:
    """How many leading words of `following` repeat the tail of `previous`.

    Chunks overlap by CHUNK_OVERLAP_MS, so the same speech is transcribed twice
    at every seam. Find the longest suffix of the previous chunk's last words
    that matches a prefix of the next chunk's first words.
    """
    if not previous or not following:
        return 0

    window = config.OVERLAP_COMPARE_WORDS
    tail: list[str] = []
    for segment in reversed(previous):
        tail = _normalize_words(segment["text"]) + tail
        if len(tail) >= window:
            break
    tail = tail[-window:]

    head: list[str] = []
    for segment in following:
        head.extend(_normalize_words(segment["text"]))
        if len(head) >= window:
            break
    head = head[:window]

    # Longest match wins: a one-word coincidence is common, a six-word one is
    # the actual seam.
    for length in range(min(len(tail), len(head)), 0, -1):
        if tail[-length:] == head[:length]:
            return length
    return 0


def _drop_leading_words(segments: list[dict], count: int) -> list[dict]:
    """Remove the first `count` words, discarding segments consumed entirely.

    `count` is measured in normalized words, because that is what
    `_overlap_word_count` compared. A token that normalizes away to nothing —
    a stray dash or ellipsis on its own — is dropped without spending one of
    those words, otherwise the two functions disagree and the seam gets trimmed
    by the wrong amount.
    """
    if count <= 0:
        return segments

    remaining = count
    for index, segment in enumerate(segments):
        tokens = segment["text"].split()
        kept_from = None
        for position, token in enumerate(tokens):
            if remaining == 0:
                kept_from = position
                break
            if _normalize_words(token):
                remaining -= 1
        if remaining:
            continue  # this whole segment was inside the overlap

        leftover = tokens[kept_from:] if kept_from is not None else []
        rest = list(segments[index + 1 :])
        if not leftover:
            return rest
        kept = segment.copy()
        kept["text"] = " ".join(leftover)
        return [kept, *rest]
    return []


def stitch(chunk_results: list[dict]) -> list[dict]:
    """Offset each chunk's timestamps into the original recording, then de-duplicate.

    `chunk_results` is [{"segments": [...], "start_ms": int}, ...] in order.
    """
    stitched: list[dict] = []
    for chunk in chunk_results:
        offset = chunk["start_ms"] / 1000.0
        segments = [
            {
                "start": segment["start"] + offset,
                "end": segment["end"] + offset,
                "text": segment["text"],
            }
            for segment in chunk.get("segments", [])
        ]
        if stitched and segments:
            repeated = _overlap_word_count(stitched, segments)
            segments = _drop_leading_words(segments, repeated)
        stitched.extend(segment for segment in segments if segment["text"].strip())
    return stitched


# --- Public entry point ---------------------------------------------------


def transcribe(
    audio_chunks: list[dict],
    progress_callback: Callable[[int, int], None] | None = None,
    language: str | None = None,
    provider: str | None = None,
) -> dict:
    """Transcribe pre-split chunks and return one continuous transcript.

    Returns {"text", "segments", "language", "duration", "failed_chunks"}, where
    segments carry timestamps relative to the start of the whole recording.

    A chunk that fails is marked inline and the rest continue — losing ten
    minutes of a meeting beats losing all of it.
    """
    provider = provider or config.ASR_PROVIDER
    if provider not in _PROVIDERS:
        raise TranscriptionError(
            f"Unknown ASR_PROVIDER {provider!r}. Expected 'groq' or 'local'."
        )
    if not audio_chunks:
        raise TranscriptionError("There is no audio to transcribe.")
    # Preflight whatever this provider needs. Setup problems fail identically on
    # every chunk, so surfacing them here keeps the specific, fixable message
    # instead of burying it under "every chunk failed".
    if provider == "groq":
        try:
            config.api_key("groq")  # fail early, before encoding anything
        except (RuntimeError, ValueError) as exc:
            raise TranscriptionError(str(exc)) from exc
    elif provider == "local" and _PROVIDERS["local"] is _transcribe_local:
        _load_local_model()

    total = len(audio_chunks)
    chunk_results = []
    detected_languages = []
    failed = 0

    for index, chunk in enumerate(audio_chunks):
        segment_audio = chunk["audio"]
        start_ms = chunk["start_ms"]
        try:
            payload = audio_module.export_bytes(segment_audio)
            result = _transcribe_chunk_with_retry(payload, language, provider)
            if result.get("language"):
                detected_languages.append(result["language"])
            chunk_results.append(
                {"segments": result["segments"], "start_ms": start_ms}
            )
        except TranscriptionError as exc:
            failed += 1
            start_s = start_ms / 1000.0
            end_s = start_s + len(segment_audio) / 1000.0
            marker = (
                f"[transcription failed for {format_timestamp(start_s)}"
                f"–{format_timestamp(end_s)}: {exc}]"
            )
            chunk_results.append(
                {
                    "segments": [{"start": 0.0, "end": end_s - start_s, "text": marker}],
                    "start_ms": start_ms,
                }
            )
        finally:
            if progress_callback:
                progress_callback(index + 1, total)

    if failed == total:
        advice = (
            "Check your GROQ_API_KEY and network connection, or set "
            "ASR_PROVIDER=local in .env to transcribe on this machine."
            if provider == "groq"
            else "Check that faster-whisper is installed and the model downloaded."
        )
        raise TranscriptionError(f"Every chunk failed to transcribe. {advice}")

    segments = stitch(chunk_results)
    last_chunk = audio_chunks[-1]
    duration = (last_chunk["start_ms"] + len(last_chunk["audio"])) / 1000.0

    return {
        "text": " ".join(segment["text"] for segment in segments).strip(),
        "segments": segments,
        "language": _dominant_language(detected_languages),
        "duration": duration,
        "failed_chunks": failed,
    }


def _dominant_language(codes: list[str]) -> str | None:
    """Most frequently detected language across chunks.

    Per-chunk detection disagrees on code-switched speech, so a majority vote
    across the whole recording is steadier than trusting the first chunk.
    """
    if not codes:
        return None
    return max(set(codes), key=codes.count)
