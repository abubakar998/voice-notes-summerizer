"""Loading, normalizing and splitting audio.

Two things here matter more than the rest of the file. Splits land in silence
rather than on a stopwatch, because cutting mid-word garbles the transcript at
every seam. And every chunk carries the offset it started at, because Whisper
timestamps each clip from zero and without the offset every timestamp after the
first chunk is wrong.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_silence

from . import config


class AudioError(RuntimeError):
    """A problem with the input audio, already phrased for a human to read."""


def _configure_pydub() -> None:
    """Point pydub at an ffmpeg binary, wherever it lives on this machine."""
    path = config.ffmpeg_path()
    if not path:
        return  # load_and_normalize raises a useful error when it's actually needed
    AudioSegment.converter = path
    AudioSegment.ffmpeg = path

    if shutil.which("ffprobe"):
        return

    # imageio-ffmpeg bundles ffmpeg but no ffprobe, and pydub shells out to
    # ffprobe on every decode — so without this, every mp3 dies on WinError 2.
    # It probes only to choose an output sample format, which is wasted work
    # here because we immediately normalize to 16-bit anyway. Returning nothing
    # makes pydub skip the flag and let ffmpeg default to pcm_s16le.
    import pydub.audio_segment

    pydub.audio_segment.mediainfo_json = lambda *args, **kwargs: {}


_configure_pydub()

_FFMPEG_MISSING = (
    "ffmpeg was not found, so anything other than a plain WAV cannot be decoded. "
    "Install it with `pip install imageio-ffmpeg`, or system-wide from "
    "https://ffmpeg.org/download.html. On Hugging Face Spaces this means the "
    "packages.txt file is missing or misnamed."
)


def load_and_normalize(file_bytes: bytes, filename: str) -> AudioSegment:
    """Decode any supported format and return 16kHz mono 16-bit audio.

    Normalizing up front is not just tidiness: it shrinks a stereo 44.1kHz
    recording by around 6x, which frequently drops a file under the API size
    limit that would otherwise have needed splitting.
    """
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        raise AudioError(
            f"That file is {size_mb:.0f}MB, over the {config.MAX_UPLOAD_MB}MB limit. "
            "Try a shorter recording, or compress it to mp3 first."
        )
    if not file_bytes:
        raise AudioError("That file is empty.")

    suffix = Path(filename).suffix.lstrip(".").lower()
    # pydub wants ffmpeg's name for the container, which differs for a couple.
    fmt = {"m4a": "mp4", "mpga": "mp3", "oga": "ogg"}.get(suffix, suffix or None)

    if fmt not in (None, "wav") and not config.ffmpeg_path():
        raise AudioError(_FFMPEG_MISSING)

    try:
        audio = AudioSegment.from_file(io.BytesIO(file_bytes), format=fmt)
    except Exception as exc:
        raise AudioError(
            f"Could not decode {filename!r}. Supported formats are mp3, wav, m4a, "
            f"ogg, webm, flac and mp4. ({exc})"
        ) from exc

    audio = (
        audio.set_frame_rate(config.TARGET_SAMPLE_RATE)
        .set_channels(config.TARGET_CHANNELS)
        .set_sample_width(config.TARGET_SAMPLE_WIDTH)
    )

    duration = get_duration_seconds(audio)
    if duration < config.MIN_AUDIO_SECONDS:
        raise AudioError(
            f"That recording is only {duration:.1f}s long. There is nothing to "
            "transcribe in under a second of audio."
        )
    # dBFS is -inf for digital silence, which compares below any threshold.
    if audio.max_dBFS < config.SILENCE_THRESHOLD_DBFS:
        raise AudioError(
            "That recording is silent. Check the microphone was actually "
            "capturing, then try again."
        )
    return audio


def get_duration_seconds(audio: AudioSegment) -> float:
    return len(audio) / 1000.0


def export_bytes(audio: AudioSegment, fmt: str = config.CHUNK_UPLOAD_FORMAT) -> bytes:
    """Encode a segment for upload. FLAC by default — lossless and compact."""
    buffer = io.BytesIO()
    try:
        audio.export(buffer, format=fmt)
    except Exception as exc:
        if fmt == "wav":
            raise AudioError(f"Could not encode audio for upload: {exc}") from exc
        # FLAC needs ffmpeg; WAV does not. Falling back keeps a machine with no
        # ffmpeg working for WAV input rather than failing at the last step.
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
    return buffer.getvalue()


def _find_cut_point(audio: AudioSegment, target_ms: int) -> int:
    """Return the best place to cut near `target_ms` — a silent gap if there is one.

    Only the window around the target is scanned, not the whole recording, so
    this stays cheap on a 60-minute file.
    """
    window = config.SILENCE_SEARCH_WINDOW_MS
    start = max(0, target_ms - window)
    end = min(len(audio), target_ms + window)
    if end - start < config.MIN_SILENCE_LEN_MS:
        return target_ms

    region = audio[start:end]
    # An absolute dBFS threshold mis-fires on quiet recordings, so derive it
    # from how loud this particular clip actually is.
    threshold = region.dBFS + config.SILENCE_DROP_DBFS
    if region.dBFS == float("-inf"):
        return target_ms

    try:
        silences = detect_silence(
            region,
            min_silence_len=config.MIN_SILENCE_LEN_MS,
            silence_thresh=threshold,
            seek_step=10,
        )
    except Exception:
        return target_ms

    if not silences:
        return target_ms  # continuous speech: a hard cut is the only option

    # Cut in the middle of whichever gap sits closest to where we wanted to cut.
    midpoints = [start + (lo + hi) // 2 for lo, hi in silences]
    return min(midpoints, key=lambda point: abs(point - target_ms))


def split_audio(audio: AudioSegment) -> list[dict]:
    """Split into overlapping chunks, each tagged with its offset in the original.

    Returns [{"audio": AudioSegment, "start_ms": int}, ...] in order. A
    recording shorter than one chunk comes back as a single chunk at offset 0.
    """
    total = len(audio)
    if total <= config.CHUNK_LENGTH_MS:
        return [{"audio": audio, "start_ms": 0}]

    boundaries = [0]
    while True:
        target = boundaries[-1] + config.CHUNK_LENGTH_MS
        if target >= total:
            break
        cut = _find_cut_point(audio, target)
        # A silence search that lands at or behind where we already are would
        # loop forever or emit an empty chunk. Fall back to the hard boundary.
        if cut <= boundaries[-1]:
            cut = target
        # Don't leave a sliver at the end; fold it into the last chunk instead.
        if total - cut < config.MIN_SILENCE_LEN_MS:
            break
        boundaries.append(cut)
    boundaries.append(total)

    chunks = []
    for index, start in enumerate(boundaries[:-1]):
        end = boundaries[index + 1]
        # Overlap into the next chunk so a word sitting on the seam appears
        # whole in at least one of them. transcribe.py trims the repeat.
        padded_end = min(total, end + config.CHUNK_OVERLAP_MS)
        chunks.append({"audio": audio[start:padded_end], "start_ms": start})
    return chunks
