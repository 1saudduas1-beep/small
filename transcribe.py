"""
Transcribes an audio file using faster-whisper (English only) and writes
the result as either:
  - srt: numbered, timed cues (one per Whisper segment) for video players
  - txt: numbered COMPLETE SENTENCES, no timestamps (one sentence per entry)

Segments are written to disk as soon as they are decoded (so progress is
visible in the log and a crash still leaves a usable partial file).

Usage:
    python transcribe.py <input_audio_path> <output_path> [format] [model_size]

    format:     "srt" (default) or "txt"
    model_size: "tiny", "base", "small" (default), "medium", "large-v2",
                "large-v3", "large-v3-turbo", "distil-large-v3"

txt sentence rules:
    * Whisper segments are merged / split so every entry ends at a sentence
      end: . ? ! … : ;  (and the Arabic ؟ ؛ ۔), optionally followed by a
      closing quote or bracket.
    * Abbreviations (Dr., Mr., e.g., i.e., U.S., vs. ...), decimals (3.5) and
      times (10:30) are not treated as sentence ends.
    * A sentence longer than MAX_SENTENCE_CHARS (text without punctuation) is
      cut at the last comma / word boundary within the limit.
"""

import os
import re
import sys

VALID_MODEL_SIZES = {
    "tiny", "base", "small", "medium", "large-v2", "large-v3",
    "large-v3-turbo", "distil-large-v3",
}
VALID_FORMATS = {"srt", "txt"}

# A real speaker almost never repeats the exact same segment more than this
# many times in a row; longer runs are Whisper hallucination loops.
MAX_CONSECUTIVE_REPEATS = 3

MAX_SENTENCE_CHARS = 300

# A terminator run (. ? ! … ؟ ۔) or a single : ; ؛, then optional closing
# quotes/brackets, and only if followed by whitespace or the end of the text
# (so 3.5, 10:30 and e.g. "U.S.A" are never matched).
SENTENCE_END_RE = re.compile(r"([.?!\u061f\u06d4\u2026]+|[:;\u061b])([\"'\u201d\u2019\u00bb)\]]*)(?=\s|$)")
LAST_WORD_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z][A-Za-z.]*)$")

# Never a sentence end when followed by a period.
HARD_ABBREVS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "mt", "vs", "fig", "vol",
    "e.g", "i.e", "u.s", "u.k",
}
# A sentence end only when the next word starts with a capital letter.
SOFT_ABBREVS = {"etc", "a.m", "p.m", "inc", "ltd", "co", "approx", "dept", "est"}


def format_timestamp(total_seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm (rounded once, in ms)."""
    total_ms = max(0, int(round(total_seconds * 1000)))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, milliseconds = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def format_entry(fmt: str, number: int, start: float, end: float, text: str) -> str:
    if fmt == "srt":
        end = max(end, start)
        return f"{number}\n{format_timestamp(start)} --> {format_timestamp(end)}\n{text}\n\n"
    return f"{number}\n{text}\n\n"


# ---------------------------------------------------------------- sentences
def _is_boundary(buf: str, m: "re.Match") -> bool:
    term = m.group(1)
    nxt = buf[m.end():].lstrip()[:1]  # next visible character ("" = end of text)

    if term == ".":
        found = LAST_WORD_RE.search(buf[:m.start(1)])
        word = found.group(1).lower() if found else ""
        if word in HARD_ABBREVS:
            return False
        if word in SOFT_ABBREVS:
            return (not nxt) or nxt.isupper()
        return True

    if term.strip(".\u2026") == "":  # ellipsis: "so... what" continues, "Well... Yes" ends
        return (not nxt) or not nxt.islower()

    return True  # ? ! : ; and Arabic marks


def _force_cut(buf: str, limit: int) -> int:
    """Where to cut an over-long sentence: last comma, else last space, else limit."""
    window = buf[:limit]
    idx = max(window.rfind(", "), window.rfind("\u060c "))
    if idx >= limit // 2:
        return idx + 1
    idx = window.rfind(" ")
    if idx > 0:
        return idx
    return limit


class SentenceSplitter:
    """Streaming splitter: feed() Whisper segment texts, get complete sentences back."""

    def __init__(self, max_chars: int = MAX_SENTENCE_CHARS):
        self.max_chars = max_chars
        self.buf = ""

    def feed(self, text: str):
        self.buf = (self.buf + " " + text).strip()
        self.buf = re.sub(r"\s+", " ", self.buf)
        return self._drain()

    def flush(self):
        rest = self.buf.strip()
        self.buf = ""
        return [rest] if rest else []

    def _drain(self):
        out = []
        while self.buf:
            cut = None
            for m in SENTENCE_END_RE.finditer(self.buf):
                if _is_boundary(self.buf, m):
                    cut = m.end()
                    break
            if cut is None:
                if len(self.buf) <= self.max_chars:
                    break
                cut = _force_cut(self.buf, self.max_chars)
            elif cut > self.max_chars:
                cut = _force_cut(self.buf, self.max_chars)
            sentence = self.buf[:cut].strip()
            self.buf = self.buf[cut:].lstrip()
            if sentence:
                out.append(sentence)
        return out


def main():
    # Unbuffered-style logging so GitHub Actions shows progress live.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if len(sys.argv) not in (3, 4, 5):
        print(
            "Usage: python transcribe.py <input_audio_path> <output_path> "
            "[srt|txt] [model_size]"
        )
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    output_format = sys.argv[3].strip().lower() if len(sys.argv) >= 4 else "srt"
    model_size = sys.argv[4].strip().lower() if len(sys.argv) == 5 else "small"

    if output_format not in VALID_FORMATS:
        print(f"ERROR: unsupported format '{output_format}'. Use 'srt' or 'txt'.", file=sys.stderr)
        sys.exit(1)

    if model_size not in VALID_MODEL_SIZES:
        print(
            f"ERROR: unsupported model_size '{model_size}'. "
            f"Choose one of: {', '.join(sorted(VALID_MODEL_SIZES))}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(input_path) or os.path.getsize(input_path) == 0:
        print(f"ERROR: input audio '{input_path}' is missing or empty.", file=sys.stderr)
        sys.exit(1)

    # Imported here so helpers above stay importable (and unit-testable)
    # without faster-whisper installed.
    from faster_whisper import WhisperModel

    # Auto-detect available CPU cores so this adapts automatically whether
    # the runner has 2 vCPUs (private repo) or 4 vCPUs (public repo).
    cpu_threads = os.cpu_count() or 4
    print(f"Loading model '{model_size}' (int8, CPU, cpu_threads={cpu_threads})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=cpu_threads)

    print(f"Transcribing '{input_path}' (language=en)...")
    segments, info = model.transcribe(
        input_path,
        language="en",
        beam_size=5,
        vad_filter=True,  # skip silence, useful for AI-generated podcasts
        vad_parameters={"min_silence_duration_ms": 500},
        # Feeding the previous text back in is the main cause of repetition
        # loops / hallucination drift in long audio; each segment is
        # decoded independently instead.
        condition_on_previous_text=False,
    )

    duration = float(info.duration or 0)
    print(f"Detected duration: {duration:.1f}s")

    splitter = SentenceSplitter(MAX_SENTENCE_CHARS) if output_format == "txt" else None
    count = 0
    skipped_repeats = 0
    last_text = None
    repeat_run = 0

    # `segments` is a lazy generator: decoding happens while we iterate, so
    # every entry is written + flushed immediately.
    with open(output_path, "w", encoding="utf-8") as f:

        def emit(start, end, text):
            nonlocal count
            count += 1
            f.write(format_entry(output_format, count, start, end, text))
            f.flush()

        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            if text == last_text:
                repeat_run += 1
            else:
                repeat_run = 1
                last_text = text
            if repeat_run > MAX_CONSECUTIVE_REPEATS:
                skipped_repeats += 1
                continue

            if splitter:
                for sentence in splitter.feed(text):
                    emit(0.0, 0.0, sentence)
            else:
                emit(segment.start, segment.end, text)

            pct = min(100.0, segment.end / duration * 100) if duration > 0 else 0.0
            print(f"  [{pct:5.1f}%] [{format_timestamp(segment.start)}] {text}")

        if splitter:
            for sentence in splitter.flush():
                emit(0.0, 0.0, sentence)

    if count == 0:
        print("ERROR: no speech segments were produced.", file=sys.stderr)
        sys.exit(1)

    if skipped_repeats:
        print(f"Note: dropped {skipped_repeats} repeated segment(s) (likely hallucination loops).")
    unit = "sentences" if splitter else "segments"
    print(f"Done. Wrote {count} {unit} to '{output_path}' (format={output_format}).")


if __name__ == "__main__":
    main()
