"""
Transcribes an audio file using faster-whisper (English only) via
BatchedInferencePipeline (batch_size=8) for faster CPU throughput at the
same accuracy as standard sequential decoding, and writes the result as
either:
  - srt: numbered segments with start/end timestamps
  - txt: numbered segments with the timestamps removed

Usage:
    python transcribe.py <input_audio_path> <output_path> [format] [model_size]

    format:     "srt" (default) or "txt"
    model_size: any faster-whisper model size, e.g. "tiny", "base",
                "small" (default), "medium", "large-v3"
"""

import os
import sys
from faster_whisper import WhisperModel, BatchedInferencePipeline

VALID_MODEL_SIZES = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}

# Documented faster-whisper CPU benchmark: batch_size=8 roughly halves
# wall-clock time versus non-batched inference at the same accuracy
# (same model weights, just parallelized decoding of VAD-split chunks).
BATCH_SIZE = 8


def format_timestamp(total_seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    if total_seconds < 0:
        total_seconds = 0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int(round((total_seconds - int(total_seconds)) * 1000))
    if milliseconds == 1000:
        milliseconds = 0
        seconds += 1
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def write_srt(segments, output_path: str) -> int:
    """Write numbered segments with timestamps (standard .srt)."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
            print(f"  [{start} --> {end}] {text}")
            count += 1
    return count


def write_txt(segments, output_path: str) -> int:
    """Write numbered segments WITHOUT timestamps."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            text = segment.text.strip()
            f.write(f"{i}\n{text}\n\n")
            print(f"  [{i}] {text}")
            count += 1
    return count


WRITERS = {
    "srt": write_srt,
    "txt": write_txt,
}


def main():
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

    if output_format not in WRITERS:
        print(f"ERROR: unsupported format '{output_format}'. Use 'srt' or 'txt'.", file=sys.stderr)
        sys.exit(1)

    if model_size not in VALID_MODEL_SIZES:
        print(
            f"ERROR: unsupported model_size '{model_size}'. "
            f"Choose one of: {', '.join(sorted(VALID_MODEL_SIZES))}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Auto-detect available CPU cores so this adapts automatically whether
    # the runner has 2 vCPUs (private repo) or 4 vCPUs (public repo),
    # without needing a code change if that ever changes.
    cpu_threads = os.cpu_count() or 4
    print(f"Loading model '{model_size}' (int8, CPU, cpu_threads={cpu_threads})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=cpu_threads)

    # BatchedInferencePipeline wraps the same model/weights and parallelizes
    # decoding of VAD-split chunks instead of processing them sequentially.
    # Accuracy is unaffected (same weights); only throughput improves.
    batched_model = BatchedInferencePipeline(model=model)

    print(f"Transcribing '{input_path}' (language=en, batch_size={BATCH_SIZE})...")
    segments, info = batched_model.transcribe(
        input_path,
        language="en",
        beam_size=5,
        vad_filter=True,  # skip silence, useful for AI-generated podcasts
        batch_size=BATCH_SIZE,
    )

    print(f"Detected duration: {info.duration:.1f}s")

    # faster-whisper returns a generator; materialize once so we can
    # both print progress and hand it to the chosen writer without
    # re-running transcription.
    segments = list(segments)

    count = WRITERS[output_format](segments, output_path)

    print(f"Done. Wrote {count} segments to '{output_path}' (format={output_format}).")


if __name__ == "__main__":
    main()
