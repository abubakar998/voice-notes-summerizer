"""Measure how well the transcription pipeline actually handles Bangla.

Anyone can wire Whisper to a summarizer. The question nobody answers is whether
it works on their language — Whisper's Bangla performance is meaningfully worse
than its English performance and is rarely quantified anywhere public.

Usage:
    python eval/run_eval.py --fetch          # build the clip set from FLEURS
    python eval/run_eval.py                  # score every system, auto vs forced Bangla
    python eval/run_eval.py --systems groq-turbo local-small
    python eval/run_eval.py --languages auto # skip the forced-language condition
    python eval/run_eval.py --report         # re-print the table from saved results

Layout, if you want to add clips by hand:
    eval/samples/<category>/<clip_id>.wav
    eval/references/<category>/<clip_id>.txt

CER matters more than WER for Bangla. Bangla is richly inflected and written
without spaces in places English would use them, so word-level scoring punishes
a nearly-right transcription far harder than a reader would.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import audio as audio_module  # noqa: E402
from src import config, transcribe  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = EVAL_DIR / "samples"
REFERENCES_DIR = EVAL_DIR / "references"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "results.json"

# Four conditions, chosen to isolate what actually degrades accuracy: the
# language itself, background noise, code-switching, and a baseline to compare
# all three against.
CATEGORIES = {
    "bangla_clean": "Clean Bangla",
    "bangla_noisy": "Bangla + noise",
    "code_switched": "Bangla-English mix",
    "english_clean": "English (baseline)",
}

# The systems compared. Hosted Whisper against local faster-whisper at two
# sizes is the comparison that tells you whether the free API is worth it.
SYSTEMS = {
    "groq-turbo": {"provider": "groq", "model": "whisper-large-v3-turbo"},
    "groq-large": {"provider": "groq", "model": "whisper-large-v3"},
    "local-base": {"provider": "local", "model": "base"},
    "local-small": {"provider": "local", "model": "small"},
}
DEFAULT_SYSTEMS = ["groq-turbo", "local-base", "local-small"]

TARGET_CLIPS_PER_CATEGORY = 7
MIN_CLIP_SECONDS = 3.0
MAX_CLIP_SECONDS = 90.0
NOISE_SNR_DB = 10.0  # a realistic noisy room, not a destroyed signal


# --- Scoring normalization ------------------------------------------------


def normalize_for_scoring(text: str) -> str:
    """Put reference and hypothesis on equal footing before comparing them.

    Unicode normalization is not optional for Bangla: the same visible
    character has several valid encodings, and without NFC two identical-looking
    strings score as completely different.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()  # affects the English only; Bangla has no case
    # Drop punctuation across scripts, including the Bangla danda (।) and
    # double danda (॥), which the Unicode 'P' categories cover.
    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char for char in text
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def score(reference: str, hypothesis: str) -> dict:
    """Word and character error rates for one clip."""
    import jiwer

    ref = normalize_for_scoring(reference)
    hyp = normalize_for_scoring(hypothesis)
    if not ref:
        return {"wer": None, "cer": None, "ref_words": 0}
    if not hyp:
        return {"wer": 1.0, "cer": 1.0, "ref_words": len(ref.split())}
    return {
        "wer": jiwer.wer(ref, hyp),
        "cer": jiwer.cer(ref, hyp),
        "ref_words": len(ref.split()),
    }


# --- Building the clip set ------------------------------------------------


def _add_noise(audio, snr_db: float = NOISE_SNR_DB):
    """Mix in white noise at a fixed signal-to-noise ratio."""
    from pydub.generators import WhiteNoise

    noise = WhiteNoise().to_audio_segment(duration=len(audio))
    noise = noise.apply_gain(audio.dBFS - noise.dBFS - snr_db)
    return audio.overlay(noise)


# Where the clips come from. Both ship a human-written sentence with every clip,
# which is what makes this practical — hand-transcribing 30 Bangla clips is a
# day's work with no shortcut.
#
# Not Common Voice, despite being the obvious choice: it is gated behind
# accepting terms with a Hugging Face token, and an evaluation nobody else can
# reproduce is worth much less. Not FLEURS either — it is ungated, but ships as a
# single 791MB parquet whose row groups are too large for the rows API, so
# collecting seven clips means downloading all of it.
#
# These two are read through Hugging Face's rows API, which serves individual
# clips as ordinary audio URLs. A few megabytes, no authentication, no bulk
# download, and no `datasets` dependency.
SOURCES = {
    "bangla_clean": {
        # Bengali.AI crowdsourced speech: Bangladeshi Bangla, which is the
        # variety this app is actually for.
        "dataset": "arif11/Bengali_AI_Speech",
        "config": "default",
        "split": "train",
        "text_field": "transcription",
    },
    "english_clean": {
        # LibriSpeech, clean read speech — the baseline everything else is
        # measured against.
        "dataset": "hf-internal-testing/librispeech_asr_dummy",
        "config": "clean",
        "split": "validation",
        "text_field": "text",
    },
}

ROWS_API = "https://datasets-server.huggingface.co/rows"


def fetch_reference_clips(limit: int = TARGET_CLIPS_PER_CATEGORY) -> None:
    """Pull clips and their reference sentences into eval/samples and eval/references."""
    for category, source in SOURCES.items():
        print(f"Fetching {limit} clips from {source['dataset']} -> {category}/ ...")
        saved = 0
        offset = 0
        while saved < limit and offset < limit * 6:  # give up rather than page forever
            try:
                rows = _fetch_rows(source, offset, batch=min(20, limit * 2))
            except Exception as exc:
                print(f"  could not reach the rows API: {exc}")
                break
            if not rows:
                break
            for row in rows:
                if saved >= limit:
                    break
                sentence = (row["row"].get(source["text_field"]) or "").strip()
                audio_url = _audio_url(row["row"])
                if not sentence or not audio_url:
                    continue
                try:
                    clip = _download_segment(audio_url)
                except Exception as exc:
                    print(f"  skipping a clip that would not decode: {exc}")
                    continue
                if not (MIN_CLIP_SECONDS <= len(clip) / 1000 <= MAX_CLIP_SECONDS):
                    continue

                clip_id = f"{category}_{saved:03d}"
                _write_clip(category, clip_id, clip, sentence)
                # Every clean Bangla clip gets a noisy twin of the same speech,
                # so the noise column measures noise alone rather than
                # confounding it with a different speaker and sentence.
                if category == "bangla_clean":
                    _write_clip("bangla_noisy", clip_id, _add_noise(clip), sentence)
                saved += 1
            offset += len(rows)
        print(f"  saved {saved} clips")

    print(
        "\nCode-switched Bangla-English speech has no usable public dataset with\n"
        "references. Record 5-8 clips into eval/samples/code_switched/ with\n"
        "matching .txt references in eval/references/code_switched/, then re-run\n"
        "without --fetch — the harness picks up whatever is there."
    )


def _fetch_rows(source: dict, offset: int, batch: int) -> list[dict]:
    import json
    import urllib.parse
    import urllib.request

    query = urllib.parse.urlencode(
        {
            "dataset": source["dataset"],
            "config": source["config"],
            "split": source["split"],
            "offset": offset,
            "length": batch,
        }
    )
    with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=60) as response:
        payload = json.loads(response.read())
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload.get("rows", [])


def _audio_url(row: dict) -> str | None:
    """The rows API returns audio as a list of {src, type} alternatives."""
    audio = row.get("audio")
    if isinstance(audio, list) and audio:
        return audio[0].get("src")
    if isinstance(audio, dict):
        return audio.get("src")
    return None


def _download_segment(url: str):
    """Fetch one clip and normalize it the same way the app would."""
    import io
    import urllib.request

    from pydub import AudioSegment

    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    return (
        AudioSegment.from_file(io.BytesIO(data))
        .set_frame_rate(config.TARGET_SAMPLE_RATE)
        .set_channels(config.TARGET_CHANNELS)
        .set_sample_width(config.TARGET_SAMPLE_WIDTH)
    )


def _write_clip(category: str, clip_id: str, clip, sentence: str) -> None:
    (SAMPLES_DIR / category).mkdir(parents=True, exist_ok=True)
    (REFERENCES_DIR / category).mkdir(parents=True, exist_ok=True)
    clip.export(SAMPLES_DIR / category / f"{clip_id}.wav", format="wav")
    (REFERENCES_DIR / category / f"{clip_id}.txt").write_text(sentence, encoding="utf-8")


# --- Running --------------------------------------------------------------


def load_clips() -> list[dict]:
    """Every clip that has a matching reference transcript."""
    clips = []
    for category in CATEGORIES:
        folder = SAMPLES_DIR / category
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.wav")):
            reference_path = REFERENCES_DIR / category / f"{path.stem}.txt"
            if not reference_path.is_file():
                print(f"  skipping {path.name}: no reference transcript")
                continue
            clips.append(
                {
                    "id": path.stem,
                    "category": category,
                    "path": path,
                    "reference": reference_path.read_text(encoding="utf-8").strip(),
                }
            )
    return clips


def transcribe_clip(clip: dict, system: str, language: str | None) -> tuple[str, float]:
    """Run one clip through one system. Returns (text, seconds taken)."""
    spec = SYSTEMS[system]
    audio = audio_module.load_and_normalize(clip["path"].read_bytes(), clip["path"].name)
    chunks = audio_module.split_audio(audio)

    # Point config at this system's model for the duration of the call.
    if spec["provider"] == "groq":
        original, config.GROQ_ASR_MODEL = config.GROQ_ASR_MODEL, spec["model"]
    else:
        original, config.LOCAL_ASR_MODEL = config.LOCAL_ASR_MODEL, spec["model"]

    started = time.time()
    try:
        result = transcribe.transcribe(chunks, provider=spec["provider"], language=language)
        return result["text"], time.time() - started
    finally:
        if spec["provider"] == "groq":
            config.GROQ_ASR_MODEL = original
        else:
            config.LOCAL_ASR_MODEL = original


def detect_script(text: str) -> str:
    """Which writing system the transcription actually came out in.

    Whisper's failure on Bangla turns out not to be sloppy spelling — it decides
    the audio is Gujarati or Tamil and transcribes it in that script, which
    scores as ~100% error no matter how good the acoustics were. Naming the
    script separates "cannot hear Bangla" from "did not realise it was Bangla".
    """
    counts: dict[str, int] = {}
    for char in text:
        if not char.strip() or unicodedata.category(char).startswith(("P", "N", "Z")):
            continue
        name = unicodedata.name(char, "")
        script = name.split()[0] if name else "UNKNOWN"
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "empty"
    return max(counts, key=counts.get).lower()


def run(systems: list[str], languages: list[str | None]) -> dict:
    clips = load_clips()
    if not clips:
        sys.exit(
            "No clips found. Run `python eval/run_eval.py --fetch` first, or add "
            "your own to eval/samples/<category>/ with references alongside."
        )

    print(f"{len(clips)} clips across {len({c['category'] for c in clips})} categories\n")
    results = []
    for system in systems:
        for language in languages:
            mode = language or "auto"
            print(f"--- {system} [{mode}] ---")
            for index, clip in enumerate(clips, start=1):
                try:
                    text, elapsed = transcribe_clip(clip, system, language)
                    measured = score(clip["reference"], text)
                    status = f"WER {measured['wer']:.1%}  CER {measured['cer']:.1%}"
                except Exception as exc:
                    text, elapsed = "", 0.0
                    measured = {"wer": None, "cer": None, "ref_words": 0}
                    status = f"FAILED: {exc}"
                print(f"  [{index}/{len(clips)}] {clip['category']}/{clip['id']}: {status}")
                results.append(
                    {
                        "system": system,
                        "language_mode": mode,
                        "clip": clip["id"],
                        "category": clip["category"],
                        "reference": clip["reference"],
                        "hypothesis": text,
                        "ref_script": detect_script(clip["reference"]),
                        "hyp_script": detect_script(text),
                        "seconds": elapsed,
                        **measured,
                    }
                )
            print()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "results": results}
    RESULTS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {RESULTS_JSON}")
    return payload


# --- Reporting ------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    usable = [value for value in values if value is not None]
    return sum(usable) / len(usable) if usable else None


def _cell(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"


def build_table(payload: dict) -> str:
    """Markdown table, ready to paste into the README."""
    results = payload["results"]
    columns = sorted(
        {(row["system"], row.get("language_mode", "auto")) for row in results},
        key=lambda pair: (list(SYSTEMS).index(pair[0]), pair[1]),
    )
    categories = [name for name in CATEGORIES if any(r["category"] == name for r in results)]

    def subset(rows, column):
        system, mode = column
        return [r for r in rows if r["system"] == system and r.get("language_mode", "auto") == mode]

    headers = [f"{system} ({mode})" for system, mode in columns]
    lines = [
        f"_Generated {payload.get('generated_at', '')}. Lower is better. "
        "`auto` lets Whisper detect the language; `bn` forces Bangla._",
        "",
        "| Condition | Clips | " + " | ".join(f"{h}<br/>WER / CER" for h in headers) + " |",
        "|---|---|" + "---|" * len(columns),
    ]
    for category in categories:
        rows = [r for r in results if r["category"] == category]
        count = len({r["clip"] for r in rows})
        cells = [
            f"{_cell(_mean([r['wer'] for r in subset(rows, c)]))} / "
            f"{_cell(_mean([r['cer'] for r in subset(rows, c)]))}"
            for c in columns
        ]
        lines.append(f"| {CATEGORIES[category]} | {count} | " + " | ".join(cells) + " |")

    speed = []
    for column in columns:
        rows = [r for r in subset(results, column) if r["seconds"]]
        speed.append(f"{_mean([r['seconds'] for r in rows]):.1f}s" if rows else "—")
    lines.append("| **Mean seconds per clip** | | " + " | ".join(speed) + " |")

    lines.extend(["", _script_summary(results)])
    return "\n".join(lines)


def _script_summary(results: list[dict]) -> str:
    """How often each system wrote Bangla audio in the Bengali script at all."""
    columns = sorted(
        {(row["system"], row.get("language_mode", "auto")) for row in results},
        key=lambda pair: (list(SYSTEMS).index(pair[0]), pair[1]),
    )
    bangla = [r for r in results if r["category"].startswith("bangla")]
    if not bangla:
        return ""

    lines = [
        "**Output written in the Bengali script** (Bangla clips only). A "
        "transcription in the wrong script scores as near-total error however "
        "clear the audio was:",
        "",
        "| System | Bengali script | Other script |",
        "|---|---|---|",
    ]
    for system, mode in columns:
        rows = [
            r
            for r in bangla
            if r["system"] == system and r.get("language_mode", "auto") == mode
        ]
        if not rows:
            continue
        correct = sum(1 for r in rows if r.get("hyp_script") == "bengali")
        wrong = ", ".join(
            sorted({r.get("hyp_script", "?") for r in rows if r.get("hyp_script") != "bengali"})
        )
        lines.append(
            f"| {system} ({mode}) | {correct}/{len(rows)} | {wrong or '—'} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="build the clip set from FLEURS")
    parser.add_argument("--report", action="store_true", help="re-print the table from saved results")
    parser.add_argument("--systems", nargs="+", choices=list(SYSTEMS), default=DEFAULT_SYSTEMS)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["auto", "bn"],
        help="'auto' to let Whisper detect, or an ISO code to force it",
    )
    parser.add_argument("--limit", type=int, default=TARGET_CLIPS_PER_CATEGORY)
    args = parser.parse_args()

    if args.fetch:
        fetch_reference_clips(args.limit)
        return

    if args.report:
        if not RESULTS_JSON.is_file():
            sys.exit(f"No saved results at {RESULTS_JSON}. Run the eval first.")
        print(build_table(json.loads(RESULTS_JSON.read_text(encoding="utf-8"))))
        return

    languages = [None if name == "auto" else name for name in args.languages]
    print(build_table(run(args.systems, languages)))


if __name__ == "__main__":
    main()
