"""Every tunable value for the app lives here.

No magic numbers anywhere else in the codebase. Environment variables are
read once, at import time, via python-dotenv.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Audio ----------------------------------------------------------------
# Whisper resamples to 16kHz mono internally no matter what you send it, so
# doing it up front is free accuracy-wise and shrinks a stereo 44.1kHz file by
# roughly 6x. That is often the difference between needing to split and not.
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # bytes; 16-bit PCM

MAX_UPLOAD_MB = 100
MIN_AUDIO_SECONDS = 1.0
# A clip whose loudest moment is below this is silence, not speech.
SILENCE_THRESHOLD_DBFS = -50.0

# --- Splitting ------------------------------------------------------------
CHUNK_LENGTH_MS = 10 * 60 * 1000  # 10 min
CHUNK_OVERLAP_MS = 3 * 1000  # 3 sec, so a word on a seam survives whole

# Look this far either side of a target boundary for a natural pause to cut on.
# Cutting mid-word garbles the text at every seam; 30s is generous enough to
# find a gap in normal speech and short enough that chunks stay near target.
SILENCE_SEARCH_WINDOW_MS = 30 * 1000
MIN_SILENCE_LEN_MS = 400
SILENCE_DROP_DBFS = -35  # relative to the clip's average loudness

# --- Transcription --------------------------------------------------------
ASR_PROVIDER = os.getenv("ASR_PROVIDER", "groq")  # "groq" | "local"

# Model IDs confirmed against provider docs on 2026-08-16.
#
# large-v3 rather than the faster large-v3-turbo, on the evidence in eval/:
# turbo identified the language of *zero* of 14 Bangla clips correctly and
# transcribed them into Gujarati, Tamil, Devanagari and Latin script, for a 97%
# character error rate. large-v3 got 13 of 14 right at 28% CER. Turbo is fine
# for English and marginally quicker, but this app is for Bangla too, and a
# transcript in the wrong alphabet is worthless however fast it arrived.
GROQ_ASR_MODEL = os.getenv("GROQ_ASR_MODEL", "whisper-large-v3")
LOCAL_ASR_MODEL = os.getenv("LOCAL_ASR_MODEL", "small")  # faster-whisper size

# Groq's free tier rejects uploads over 25MB (dev tier gets 100MB). We send
# FLAC: lossless, natively supported, and about half the size of the equivalent
# WAV, which keeps a 10-minute 16kHz mono chunk near 10MB with room to spare.
GROQ_MAX_FILE_MB = 25
CHUNK_UPLOAD_FORMAT = "flac"

# Whisper hallucinates confidently on silence. Auto-detect is the default
# because hardcoding a language is wrong for a bilingual app, but short or
# heavily code-switched clips detect badly, so the UI can override this.
ASR_LANGUAGE: str | None = None

# --- Summarization --------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" | "groq"

# Groq handles transcription and Gemini handles summarization by default, so a
# rate limit on one does not take the whole app down with it.
LLM_MODELS = {
    "gemini": os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
    "groq": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
}
LLM_MODEL = LLM_MODELS.get(LLM_PROVIDER, LLM_MODELS["gemini"])

# Above this, summarize in parts and merge. Roughly an hour of speech; well
# inside Gemini Flash's context window, but map-reduce produces a better
# summary than one enormous prompt, which tends to over-weight the opening.
SUMMARY_INPUT_LIMIT_CHARS = 30000
MAP_CHUNK_CHARS = 12000
MAP_CHUNK_OVERLAP_CHARS = 500

MAX_TOKENS = 2048
TEMPERATURE = 0.2  # summarize, don't embellish

# 429s are normal on a free tier. Back off 1s, 2s, 4s, 8s, then give up.
RETRY_BACKOFF_SECONDS = [1, 2, 4, 8]

# --- De-duplication -------------------------------------------------------
# Overlapping chunks repeat words at each seam. Compare this many words either
# side of a join to find the repeat and trim it.
OVERLAP_COMPARE_WORDS = 15

# --- API keys -------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

_KEY_HELP = {
    "gemini": (
        "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
        "and put it in a .env file at the project root (see .env.example)."
    ),
    "groq": (
        "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
        "and put it in a .env file at the project root (see .env.example)."
    ),
}


def api_key(provider: str) -> str:
    """Return the API key for `provider`, raising a readable error if absent."""
    key = {"gemini": GEMINI_API_KEY, "groq": GROQ_API_KEY}.get(provider)
    if not key:
        if provider not in _KEY_HELP:
            raise ValueError(
                f"Unknown provider {provider!r}. Expected 'gemini' or 'groq'."
            )
        raise RuntimeError(_KEY_HELP[provider])
    return key


def has_api_key(provider: str) -> bool:
    """True if a key for `provider` is configured. Never raises."""
    return bool({"gemini": GEMINI_API_KEY, "groq": GROQ_API_KEY}.get(provider))


def ffmpeg_path() -> str | None:
    """Locate an ffmpeg binary, or None if there genuinely isn't one.

    Prefers a real system install (deploy platforms install one from
    packages.txt), and falls back to the static binary imageio-ffmpeg bundles
    so a clean clone works on a machine with nothing installed.
    """
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
