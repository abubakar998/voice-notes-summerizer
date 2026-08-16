"""Transcript → structured summary, with map-reduce for long recordings.

Gemini handles this by default while Groq handles transcription, so the two
halves of the pipeline draw on separate free-tier quotas and a limit on one
does not take the whole app down.

Long transcripts are summarized in parts and merged rather than truncated. A
user who uploads an hour-long meeting and gets a summary of the first ten
minutes has been handed something worse than nothing, because it looks complete.
"""

from __future__ import annotations

import re
import time
from typing import Callable, Iterator

from . import config

SUMMARY_PROMPT = """You are summarizing a voice recording transcript.

The transcript may contain speech recognition errors, filler words,
and false starts. Read past them.

Produce:

## Summary
2-4 sentences on what this recording is about.

## Key Points
Bulleted. Only points actually stated in the transcript.

## Action Items
Bulleted, each with an owner if one is named. Write "None identified"
if there are none. Do not invent tasks.

If the transcript is in Bangla, write the summary in Bangla.
If it mixes Bangla and English, use whichever language dominates.

<transcript>
{transcript}
</transcript>"""

MAP_PROMPT = """This is part {index} of {total} of a longer voice recording transcript.

Summarize only this part. It may begin or end mid-sentence — that is expected,
so do not remark on it. The transcript may contain speech recognition errors
and filler words; read past them.

Write:
- A short paragraph on what is discussed in this part.
- A bulleted list of the specific points made.
- A bulleted list of any action items, with an owner if one is named.

Keep every detail that a reader of the whole recording would want. Do not
invent anything that is not stated.

Write in the same language as the transcript.

<transcript_part>
{transcript}
</transcript_part>"""

REDUCE_PROMPT = """Below are summaries of consecutive parts of one voice recording,
in order. Merge them into a single summary of the whole recording.

Merge carefully:
- Combine points that repeat across parts into one, keeping the clearest wording.
- Preserve every distinct action item. Never drop one.
- Keep the chronological sense of how the recording developed.

Produce:

## Summary
2-4 sentences on what this recording is about, as a whole.

## Key Points
Bulleted. Only points actually stated in the part summaries.

## Action Items
Bulleted, each with an owner if one is named. Write "None identified"
if there are none. Do not invent tasks.

Write in the same language as the part summaries below.

<part_summaries>
{summaries}
</part_summaries>"""

RATE_LIMIT_MESSAGE = (
    "The model provider is rate limiting us (free tier). Wait a few seconds and "
    "try again, or switch LLM_PROVIDER in src/config.py to the other provider."
)

BUSY_MESSAGE = (
    "The model provider is temporarily overloaded and did not recover after "
    "several retries. Wait a minute and try again, or switch LLM_PROVIDER in "
    "src/config.py to the other provider."
)


class SummaryError(RuntimeError):
    """A provider failure, already phrased for a human to read."""


def _is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return (
        "429" in text
        or "rate limit" in text
        or "resource_exhausted" in text
        or "quota" in text
    )


def _is_transient(exc: Exception) -> bool:
    """Worth retrying: throttling, or the provider briefly falling over.

    Free tiers serve 503 "high demand" as routinely as they serve 429, and a
    map-reduce over an hour-long transcript makes enough calls to hit one. Both
    deserve the same backoff — treating only 429 as retryable means a single
    blip throws away every part summary computed so far.
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


# --- Providers ------------------------------------------------------------


def _gemini_stream(prompt: str) -> Iterator[str]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.api_key("gemini"))
    stream = client.models.generate_content_stream(
        model=config.LLM_MODELS["gemini"],
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            # We pass no tools, so turn AFC off — otherwise the SDK logs a
            # warning about it on every single streamed call.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def _groq_stream(prompt: str) -> Iterator[str]:
    from groq import Groq

    client = Groq(api_key=config.api_key("groq"))
    stream = client.chat.completions.create(
        model=config.LLM_MODELS["groq"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


_PROVIDERS = {"gemini": _gemini_stream, "groq": _groq_stream}

# The other provider to try when the configured one is exhausted or down.
_FALLBACK = {"gemini": "groq", "groq": "gemini"}


def _stream_with_fallback(
    prompt: str,
    provider: str,
    notice: Callable[[str], None] | None = None,
) -> Iterator[str]:
    """Stream from `provider`, failing over to the other one if it is unusable.

    Splitting transcription and summarization across two providers only buys
    resilience if something actually crosses over when one goes down. A user on
    the deployed demo cannot edit config.py, so "switch LLM_PROVIDER" is not a
    recovery path for them — this is.
    """
    try:
        yield from _stream(prompt, provider)
        return
    except SummaryError as exc:
        alternative = _FALLBACK.get(provider)
        if not alternative or not config.has_api_key(alternative):
            raise
        # Only a busy or throttled provider is worth failing over. A malformed
        # request would fail identically on the other one.
        if not any(hint in str(exc) for hint in (RATE_LIMIT_MESSAGE, BUSY_MESSAGE)):
            raise
        if notice:
            notice(
                f"{provider} was unavailable, so the summary came from "
                f"{alternative} ({config.LLM_MODELS[alternative]}) instead."
            )
        yield from _stream(prompt, alternative)


def _stream(prompt: str, provider: str) -> Iterator[str]:
    """Stream one completion, retrying transient failures with exponential backoff.

    Once any text has been yielded a retry would duplicate output, so past that
    point we surface the error instead of retrying.
    """
    stream_fn = _PROVIDERS[provider]
    for delay in [*config.RETRY_BACKOFF_SECONDS, None]:
        emitted = False
        try:
            for piece in stream_fn(prompt):
                emitted = True
                yield piece
            return
        except Exception as exc:
            if emitted:
                raise SummaryError(
                    f"The connection to the model dropped mid-summary: {exc}"
                ) from exc
            if _is_transient(exc) and delay is not None:
                time.sleep(delay)
                continue
            if _is_transient(exc):
                raise SummaryError(
                    RATE_LIMIT_MESSAGE if _is_rate_limit(exc) else BUSY_MESSAGE
                ) from exc
            raise SummaryError(f"The model provider returned an error: {exc}") from exc


def _complete(
    prompt: str, provider: str, notice: Callable[[str], None] | None = None
) -> str:
    """Non-streaming completion, for the map steps nobody watches."""
    return "".join(_stream_with_fallback(prompt, provider, notice)).strip()


# --- Map-reduce -----------------------------------------------------------


def split_for_map(text: str) -> list[str]:
    """Split a long transcript into overlapping parts, preferring sentence ends.

    Overlap carries a little context across the join so a point made across a
    boundary is not lost from both parts.
    """
    limit = config.MAP_CHUNK_CHARS
    if len(text) <= limit:
        return [text]

    parts = []
    start = 0
    while start < len(text):
        end = start + limit
        if end >= len(text):
            parts.append(text[start:])
            break
        # Back up to the last sentence ending in the final quarter of the window.
        # '।' is the Bangla full stop, which the English boundaries miss.
        window = text[start + (limit * 3 // 4) : end]
        match = None
        for match in re.finditer(r"[.!?।\n]\s", window):
            pass  # keep the last one
        if match:
            end = start + (limit * 3 // 4) + match.end()
        parts.append(text[start:end])
        start = max(end - config.MAP_CHUNK_OVERLAP_CHARS, start + 1)
    return [part for part in parts if part.strip()]


def summarize(
    transcript: str,
    provider: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    notice_callback: Callable[[str], None] | None = None,
) -> Iterator[str]:
    """Yield the summary in pieces, so the UI can render it as it arrives.

    Short transcripts take one call. Long ones are summarized part by part and
    merged, with only the merge streamed. If the configured provider is down,
    the work moves to the other one and `notice_callback` is told about it.
    """
    provider = provider or config.LLM_PROVIDER
    if provider not in _PROVIDERS:
        raise SummaryError(
            f"Unknown LLM provider {provider!r}. Expected 'gemini' or 'groq'."
        )
    if not transcript or not transcript.strip():
        raise SummaryError("There is no transcript to summarize.")

    try:
        config.api_key(provider)  # fail loudly and early on a missing key
    except (RuntimeError, ValueError) as exc:
        raise SummaryError(str(exc)) from exc

    # Fire the fallback notice at most once, however many calls trip over it.
    announced: list[str] = []

    def notice(message: str) -> None:
        if notice_callback and not announced:
            announced.append(message)
            notice_callback(message)

    if len(transcript) <= config.SUMMARY_INPUT_LIMIT_CHARS:
        if progress_callback:
            progress_callback(1, 1)
        yield from _stream_with_fallback(
            SUMMARY_PROMPT.format(transcript=transcript), provider, notice
        )
        return

    parts = split_for_map(transcript)
    summaries = []
    for index, part in enumerate(parts, start=1):
        prompt = MAP_PROMPT.format(index=index, total=len(parts), transcript=part)
        summaries.append(
            f"--- Part {index} of {len(parts)} ---\n{_complete(prompt, provider, notice)}"
        )
        if progress_callback:
            progress_callback(index, len(parts) + 1)

    if progress_callback:
        progress_callback(len(parts) + 1, len(parts) + 1)
    yield from _stream_with_fallback(
        REDUCE_PROMPT.format(summaries="\n\n".join(summaries)), provider, notice
    )


def summarize_text(
    transcript: str,
    provider: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    notice_callback: Callable[[str], None] | None = None,
) -> str:
    """Collect the whole summary into a string. For scripts and tests."""
    return "".join(
        summarize(
            transcript,
            provider=provider,
            progress_callback=progress_callback,
            notice_callback=notice_callback,
        )
    ).strip()
