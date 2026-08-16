"""Streamlit UI. Deliberately thin — all the logic lives in src/ so it can be
tested without launching a browser."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime

import streamlit as st

from src import audio as audio_module
from src import config, formatting
from src import summarize as summarize_module
from src import transcribe as transcribe_module

st.set_page_config(
    page_title="Voice notes → summary",
    page_icon=":material/graphic_eq:",
    layout="centered",
)

# Whisper auto-detects, but it detects badly on short clips and on heavily
# code-switched speech, so the user can force the issue.
LANGUAGE_OPTIONS = {
    "Auto-detect": None,
    "Bangla": "bn",
    "English": "en",
}

RESULT_KEYS = ("transcript", "summary", "metadata", "processed_hash")

for key in RESULT_KEYS:
    st.session_state.setdefault(key, None)
st.session_state.setdefault("audio_bytes", None)
st.session_state.setdefault("filename", None)
st.session_state.setdefault("audio_hash", None)


@st.cache_resource(max_entries=1, show_spinner=False)
def load_audio(digest: str, _audio_bytes: bytes, _filename: str):
    """Decode, normalize and split once, keyed on the audio's hash.

    The underscore-prefixed arguments are skipped when computing the cache key,
    so `digest` alone identifies the entry and a 60-minute file is not hashed
    again on every rerun. Splitting is cached alongside decoding because
    scanning for silence at each boundary is not free either, and both the
    preview and the run itself need the result.
    """
    audio = audio_module.load_and_normalize(_audio_bytes, _filename)
    return audio, audio_module.split_audio(audio)


def clear_results() -> None:
    for key in RESULT_KEYS:
        st.session_state[key] = None


def start_over() -> None:
    clear_results()
    st.session_state.audio_bytes = None
    st.session_state.filename = None
    st.session_state.audio_hash = None
    load_audio.clear()


# --- Sidebar --------------------------------------------------------------

with st.sidebar:
    st.subheader("Recording")

    uploaded = st.file_uploader(
        "Upload audio",
        type=["mp3", "wav", "m4a", "ogg", "webm", "flac", "mp4"],
        help=f"Up to {config.MAX_UPLOAD_MB}MB. Longer recordings are split automatically.",
    )
    st.caption("or record straight into the browser")
    recorded = st.audio_input("Record a voice note")

    language_label = st.selectbox(
        "Language",
        list(LANGUAGE_OPTIONS),
        help=(
            "Auto-detect works well on clear speech. Set it explicitly for short "
            "clips or heavy Bangla-English mixing, where detection is unreliable."
        ),
    )
    language_override = LANGUAGE_OPTIONS[language_label]

    st.divider()
    with st.container(border=True):
        st.caption("Pipeline")
        asr_model = (
            config.GROQ_ASR_MODEL
            if config.ASR_PROVIDER == "groq"
            else f"faster-whisper {config.LOCAL_ASR_MODEL}"
        )
        st.markdown(f"**Transcription** · {config.ASR_PROVIDER}\n\n`{asr_model}`")
        st.markdown(f"**Summary** · {config.LLM_PROVIDER}\n\n`{config.LLM_MODEL}`")

        missing = [
            name
            for name, provider in (("GROQ_API_KEY", "groq"), ("GEMINI_API_KEY", "gemini"))
            if not config.has_api_key(provider)
        ]
        if missing:
            st.warning(f"Not set: {', '.join(missing)}", icon=":material/key_off:")
        else:
            st.caption(":material/check_circle: API keys loaded")

    st.button("Start over", icon=":material/restart_alt:", on_click=start_over)


# --- Input handling -------------------------------------------------------

st.title("Voice notes → transcript + summary")
st.caption(
    "Upload or record audio and get a timestamped transcript plus a structured "
    "summary. Nothing is stored — everything disappears when you close the tab."
)

source = uploaded or recorded
if source is not None:
    audio_bytes = source.getvalue()
    digest = hashlib.sha256(audio_bytes).hexdigest()
    # Same audio submitted again (a rerun, or a re-upload) must not retrigger a
    # 60-second job.
    if digest != st.session_state.audio_hash:
        clear_results()
        st.session_state.audio_bytes = audio_bytes
        st.session_state.filename = getattr(source, "name", None) or "recording.wav"
        st.session_state.audio_hash = digest

if st.session_state.audio_bytes is None:
    st.info(
        "Upload a file or record a voice note in the sidebar to begin.",
        icon=":material/mic:",
    )
    st.stop()


# --- Preview --------------------------------------------------------------

st.audio(st.session_state.audio_bytes)

try:
    audio, chunks = load_audio(
        st.session_state.audio_hash,
        st.session_state.audio_bytes,
        st.session_state.filename,
    )
except audio_module.AudioError as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

duration = audio_module.get_duration_seconds(audio)
chunk_count = len(chunks)

with st.container(horizontal=True):
    st.metric("Length", formatting.format_duration(duration))
    st.metric("Segments", chunk_count)
    st.metric("File", st.session_state.filename, help=st.session_state.filename)

already_done = st.session_state.processed_hash == st.session_state.audio_hash
run = st.button(
    "Transcribe and summarize",
    icon=":material/auto_awesome:",
    type="primary",
    disabled=already_done,
    help="Already processed — hit Start over to run it again." if already_done else None,
)


# --- Processing -----------------------------------------------------------

if run:
    started = time.time()
    summary_slot = st.container()

    try:
        with st.status("Preparing audio", expanded=True) as status:
            st.write(
                f"Normalized to {config.TARGET_SAMPLE_RATE // 1000}kHz mono · "
                f"{formatting.format_duration(duration)} · {chunk_count} segment(s)"
            )

            status.update(label=f"Transcribing 1 of {chunk_count}")
            progress = st.progress(0.0)

            def on_transcribe_progress(current: int, total: int) -> None:
                progress.progress(current / total)
                status.update(label=f"Transcribing {min(current + 1, total)} of {total}")

            result = transcribe_module.transcribe(
                chunks,
                progress_callback=on_transcribe_progress,
                language=language_override,
            )

            detected = formatting.language_name(result["language"])
            words = formatting.count_words(result["text"])
            st.write(f"Transcribed {words} words · detected {detected}")
            if result["failed_chunks"]:
                st.write(
                    f":material/warning: {result['failed_chunks']} segment(s) failed "
                    "and are marked in the transcript."
                )

            status.update(label="Summarizing", state="running")
            summary_progress = st.progress(0.0)

            def on_summary_progress(current: int, total: int) -> None:
                summary_progress.progress(current / total)
                if total > 1:
                    status.update(label=f"Summarizing part {current} of {total}")

            notices: list[str] = []

            def on_notice(message: str) -> None:
                notices.append(message)
                st.write(f":material/swap_horiz: {message}")

            summary_stream = summarize_module.summarize(
                result["text"],
                progress_callback=on_summary_progress,
                notice_callback=on_notice,
            )
            with summary_slot:
                st.subheader("Summary")
                summary_text = st.write_stream(summary_stream)

            status.update(label="Done", state="complete", expanded=False)

    except (audio_module.AudioError, transcribe_module.TranscriptionError) as exc:
        st.error(str(exc), icon=":material/error:")
        st.stop()
    except summarize_module.SummaryError as exc:
        # The transcript is the expensive half and it succeeded — keep it.
        st.session_state.transcript = result["segments"]
        st.session_state.summary = ""
        st.session_state.metadata = {
            "filename": st.session_state.filename,
            "duration": result["duration"],
            "language": result["language"],
            "word_count": formatting.count_words(result["text"]),
            "processing_seconds": time.time() - started,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "asr_model": asr_model,
            "llm_model": config.LLM_MODEL,
        }
        st.session_state.processed_hash = st.session_state.audio_hash
        st.warning(
            f"Transcript is ready, but the summary failed: {exc}",
            icon=":material/warning:",
        )
        st.rerun()

    st.session_state.transcript = result["segments"]
    st.session_state.summary = summary_text
    st.session_state.metadata = {
        "filename": st.session_state.filename,
        "duration": result["duration"],
        "language": result["language"],
        "word_count": formatting.count_words(result["text"]),
        "processing_seconds": time.time() - started,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "asr_model": asr_model,
        "llm_model": config.LLM_MODEL,
        "notices": notices,
    }
    st.session_state.processed_hash = st.session_state.audio_hash
    # Re-render from session state so there is exactly one code path that
    # draws results, whether they were just produced or survived a rerun.
    st.rerun()


# --- Results --------------------------------------------------------------

if st.session_state.processed_hash != st.session_state.audio_hash:
    st.stop()

metadata = st.session_state.metadata
transcript = st.session_state.transcript or []

st.divider()
with st.container(horizontal=True):
    st.metric("Duration", formatting.format_duration(metadata["duration"]))
    st.metric("Language", formatting.language_name(metadata["language"]))
    st.metric("Words", f"{metadata['word_count']:,}")
    st.metric("Processed in", f"{metadata['processing_seconds']:.0f}s")

for message in metadata.get("notices") or []:
    st.info(message, icon=":material/swap_horiz:")

st.subheader("Summary")
st.markdown(st.session_state.summary or "_The summary step failed for this recording._")

# The transcript is long and the summary is what people came for, so it starts
# collapsed underneath.
with st.expander(f"Full transcript · {len(transcript)} segments"):
    st.text(formatting.format_transcript(transcript))

stem = (st.session_state.filename or "voice-note").rsplit(".", 1)[0]
with st.container(horizontal=True):
    st.download_button(
        "Download .md",
        data=formatting.to_markdown(transcript, st.session_state.summary or "", metadata),
        file_name=f"{stem}-summary.md",
        mime="text/markdown",
        icon=":material/download:",
    )
    st.download_button(
        "Download .txt",
        data=formatting.to_plaintext(transcript, st.session_state.summary or "", metadata),
        file_name=f"{stem}-summary.txt",
        mime="text/plain",
        icon=":material/download:",
    )
