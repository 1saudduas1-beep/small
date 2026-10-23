"""
Transcribes an audio file using faster-whisper (English only) and writes
the result as either:
  - srt: numbered, timed cues for video players. Cues are built from WORD
         timestamps, so every cue starts when its first word is spoken and
         ends when its last word ends (timing follows the speech; a cue never
         waits for a sentence to finish)
  - txt: numbered COMPLETE SENTENCES, no timestamps (one sentence per entry)

Segments are written to disk as soon as they are decoded (so progress is
visible in the log and a crash still leaves a usable partial file).

Usage:
    python transcribe.py <input_audio_path> <output_path> [format] [model_size]

    format:     "srt" (default) or "txt"
    model_size: "tiny", "base", "small" (default), "medium", "large-v2",
                "large-v3", "large-v3-turbo", "distil-large-v3"

srt cue rules (word-timestamp based):
    * A cue is closed at a pause between words (>= PAUSE_BREAK_SECONDS), when
      it reaches MAX_CUE_SECONDS / MAX_CUE_CHARS, or at a comma / sentence end
      once it is long enough (SOFT_BREAK_*). Cues never overlap.
    * If a segment has no word timestamps, its text is spread over the
      segment duration in proportion to character count.

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

# ---- srt cue building (time-proportional, word-timestamp based)
PAUSE_BREAK_SECONDS = 0.45     # silence between two words that closes the cue
MAX_CUE_SECONDS = 5.0          # hard cap on cue duration
MAX_CUE_CHARS = 80             # hard cap on cue length (one comfortable line pair)
SOFT_MIN_SECONDS = 1.0         # a sentence end closes the cue once it lasts this long
SOFT_COMMA_SECONDS = 2.0       # a comma / ; : closes the cue once it lasts this long
SOFT_COMMA_CHARS = 45          # ...or once it has this many characters
END_HOLD_SECONDS = 0.12        # small hold after the last word (clipped to next cue)
MIN_CUE_SECONDS = 0.30         # shortest displayed cue (clipped to next cue)
CLAUSE_END_RE = re.compile(r"[,;:\u060c\u061b][\"'\u201d\u2019\u00bb)\]]*$")
SENT_END_RE = re.compile(r"[.?!\u061f\u06d4\u2026][\"'\u201d\u2019\u00bb)\]]*$")

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


# --------------------------------------------------------------------- cues
def _segment_words(segment):
    """[(start, end, text)] for a segment: real word timings when available,
    else the segment text spread over its duration in proportion to characters."""
    out = []
    for w in (getattr(segment, "words", None) or []):
        t = (w.word or "").strip()
        if t:
            out.append((float(w.start), float(w.end), t))
    if out:
        return out
    tokens = segment.text.split()
    if not tokens:
        return []
    start, end = float(segment.start), float(segment.end)
    span = max(end - start, 0.0)
    total = sum(len(t) for t in tokens) or 1
    pos = start
    for t in tokens:
        step = span * len(t) / total
        out.append((pos, pos + step, t))
        pos += step
    return out


class CueBuilder:
    """Streaming builder: feed() words with their own timings, get timed cues
    back as (start, end, text). A cue starts at its first word's start and ends
    at its last word's end, so subtitles follow the speech instead of waiting
    for a sentence to complete."""

    def __init__(self):
        self.words = []          # [(start, end, text)]
        self.last_end = 0.0      # end of the last closed cue (no-overlap guard)
        self.hold = None         # last closed cue, held back one step to absorb orphans

    def _chars(self, extra: str = "") -> int:
        n = sum(len(w[2]) for w in self.words) + max(len(self.words) - 1, 0)
        return n + (1 + len(extra) if self.words else len(extra))

    def _close(self, next_start=None):
        if not self.words:
            return None
        start = max(self.words[0][0], self.last_end)
        end = max(self.words[-1][1], start)
        end = max(end + END_HOLD_SECONDS, start + MIN_CUE_SECONDS) if end - start < MIN_CUE_SECONDS \
            else end + END_HOLD_SECONDS
        if next_start is not None:
            end = min(end, max(next_start, start))
        text = re.sub(r"\s+", " ", " ".join(w[2] for w in self.words)).strip()
        self.words = []
        self.last_end = end
        return (start, end, text) if text else None

    def _push(self, cue, out):
        """Hold each closed cue one step; a 1-2 word / <1 s orphan that follows
        without a pause is merged into the held cue instead of standing alone."""
        h = self.hold
        if h and cue:
            n_words = len(cue[2].split())
            joined = h[2] + " " + cue[2]
            if (n_words <= 2 and cue[1] - cue[0] < 1.0 and cue[0] - h[1] <= 0.2
                    and len(joined) <= MAX_CUE_CHARS + 15
                    and cue[1] - h[0] <= MAX_CUE_SECONDS + 1.0):
                self.hold = (h[0], cue[1], joined)
                return
        if h:
            out.append(h)
        self.hold = cue

    def feed(self, words):
        out = []
        for ws, we, wt in words:
            ws = max(ws, self.words[-1][1] if self.words else self.last_end)
            we = max(we, ws)
            if self.words:
                gap = ws - self.words[-1][1]
                dur_with = we - self.words[0][0]
                if (gap >= PAUSE_BREAK_SECONDS
                        or dur_with > MAX_CUE_SECONDS
                        or self._chars(wt) > MAX_CUE_CHARS):
                    self._push(self._close(next_start=ws), out)
            self.words.append((ws, we, wt))
            dur = we - self.words[0][0]
            if SENT_END_RE.search(wt) and dur >= SOFT_MIN_SECONDS:
                pass_break = True
            elif CLAUSE_END_RE.search(wt) and (dur >= SOFT_COMMA_SECONDS
                                               or self._chars() >= SOFT_COMMA_CHARS):
                pass_break = True
            else:
                pass_break = False
            if pass_break:
                self._push(self._close(), out)
        return out

    def flush(self):
        out = []
        self._push(self._close(), out)
        if self.hold:
            out.append(self.hold)
            self.hold = None
        return out


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
        # srt: per-word timings drive the cues (time-proportional subtitles).
        word_timestamps=(output_format == "srt"),
        hallucination_silence_threshold=(2.0 if output_format == "srt" else None),
        # Feeding the previous text back in is the main cause of repetition
        # loops / hallucination drift in long audio; each segment is
        # decoded independently instead.
        condition_on_previous_text=False,
    )

    duration = float(info.duration or 0)
    print(f"Detected duration: {duration:.1f}s")

    splitter = SentenceSplitter(MAX_SENTENCE_CHARS) if output_format == "txt" else None
    cues = CueBuilder() if output_format == "srt" else None
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
                for c_start, c_end, c_text in cues.feed(_segment_words(segment)):
                    emit(c_start, c_end, c_text)

            pct = min(100.0, segment.end / duration * 100) if duration > 0 else 0.0
            print(f"  [{pct:5.1f}%] [{format_timestamp(segment.start)}] {text}")

        if splitter:
            for sentence in splitter.flush():
                emit(0.0, 0.0, sentence)
        if cues:
            for c_start, c_end, c_text in cues.flush():
                emit(c_start, c_end, c_text)

    if count == 0:
        print("ERROR: no speech segments were produced.", file=sys.stderr)
        sys.exit(1)

    if skipped_repeats:
        print(f"Note: dropped {skipped_repeats} repeated segment(s) (likely hallucination loops).")
    unit = "sentences" if splitter else "cues"
    print(f"Done. Wrote {count} {unit} to '{output_path}' (format={output_format}).")


if __name__ == "__main__":
    main()
