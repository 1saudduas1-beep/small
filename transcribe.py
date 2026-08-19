"""
Transcribes an audio file into an SRT subtitle file (with per-segment
timestamps) using faster-whisper, model size "small", English only.

Usage:
    python transcribe.py <input_audio_path> <output_srt_path>
"""

import sys
from faster_whisper import WhisperModel


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


def main():
    if len(sys.argv) != 3:
        print("Usage: python transcribe.py <input_audio_path> <output_srt_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading model 'small' (int8, CPU)...")
    model = WhisperModel("small", device="cpu", compute_type="int8")

    print(f"Transcribing '{input_path}' (language=en)...")
    segments, info = model.transcribe(
        input_path,
        language="en",
        beam_size=5,
        vad_filter=True,  # skip silence, useful for AI-generated podcasts
    )

    print(f"Detected duration: {info.duration:.1f}s")

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            text = segment.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
            count += 1
            print(f"  [{start} --> {end}] {text}")

    print(f"Done. Wrote {count} segments to '{output_path}'.")


if __name__ == "__main__":
    main()
