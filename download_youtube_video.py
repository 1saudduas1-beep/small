"""
Downloads ONE YouTube video (video + audio, merged into a phone-friendly .mp4)
with yt-dlp, reusing the helpers of download_youtube_audio.py.

Usage:
    python download_youtube_video.py <youtube_url> <output_stem> <result_file> [max_height]

    The video is saved as <output_stem>.mp4 and its file name is written to
    <result_file>. max_height (default 240) caps the resolution: the best
    available stream up to that height is used, H.264/AAC preferred so the file
    plays on any phone.

Requirements on the runner: yt-dlp[default], Deno, ffmpeg/ffprobe (the workflow
installs them). Cookies: same secrets as the audio downloader.

Optional env vars:
    YOUTUBE_COOKIES / YOUTUBE_COOKIES_B64   see download_youtube_audio.py
    YOUTUBE_MAX_MINUTES                     refuse longer videos (default 240)
    YOUTUBE_VIDEO_MAX_MB                    refuse larger files (default 1500)
"""

import glob
import os
import shutil
import sys
import time

import download_youtube_audio as base

DEFAULT_HEIGHT = 240
ALLOWED_HEIGHTS = (144, 240, 360, 480, 720, 1080, 1440, 2160)
DEFAULT_MAX_MB = 1500


def download_video(url, stem, cookie_path, max_minutes, height, max_mb):
    import yt_dlp

    if not shutil.which("deno"):
        print("WARNING: no 'deno' on PATH; YouTube extraction may fail without a JavaScript runtime.")
    if not shutil.which("ffmpeg"):
        raise base.UserFacingError("ffmpeg is required to merge video and audio.")

    for stale in glob.glob(glob.escape(stem) + ".*"):
        try:
            os.remove(stale)
        except OSError:
            pass

    opts = {
        "format": f"bv*[height<={height}]+ba/b[height<={height}]",
        # H.264 + AAC first (plays everywhere), then the highest resolution allowed.
        "format_sort": ["vcodec:h264", "res", "acodec:aac"],
        "merge_output_format": "mp4",
        "outtmpl": {"default": f"{stem}.%(ext)s"},
        "noplaylist": True,
        "restrictfilenames": True,
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 3,
        "socket_timeout": 30,
        "noprogress": True,
        "max_filesize": int(max_mb * 1_048_576),
    }
    if cookie_path:
        opts["cookiefile"] = cookie_path

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
            raise base.UserFacingError("Live streams / upcoming premieres are not supported.")
        duration = info.get("duration")
        if duration and duration > max_minutes * 60:
            raise base.UserFacingError(
                f"The video is {duration / 60:.0f} min long; the limit is {max_minutes:.0f} min "
                "(raise YOUTUBE_MAX_MINUTES if you really need it)."
            )
        title = " ".join(str(info.get("title") or "").split())[:120]
        print(f"Video: {title} | duration: {(duration or 0) / 60:.1f} min | max height: {height}p")
        result = ydl.process_ie_result(info, download=True)

    path = base.pick_output(result, stem)
    if not path or not path.lower().endswith(".mp4"):
        raise base.UserFacingError(
            f"No .mp4 was produced (the file may exceed {max_mb:.0f} MB; pick a lower quality).",
            retryable=False,
        )
    actual = base.probe_duration(path)
    if duration and actual is not None and abs(actual - duration) > max(5.0, 0.02 * duration):
        os.remove(path)
        raise base.UserFacingError(
            f"Downloaded video is {actual:.0f}s but the source is {duration:.0f}s (partial download).",
            retryable=True,
        )
    return path, actual


def main():
    if len(sys.argv) not in (4, 5):
        print("Usage: python download_youtube_video.py <youtube_url> <output_stem> <result_file> [max_height]",
              file=sys.stderr)
        sys.exit(1)
    url, stem, result_file = sys.argv[1:4]
    try:
        height = int(sys.argv[4]) if len(sys.argv) == 5 else DEFAULT_HEIGHT
    except ValueError:
        height = DEFAULT_HEIGHT
    if height not in ALLOWED_HEIGHTS:
        print(f"ERROR: max_height must be one of {ALLOWED_HEIGHTS}.", file=sys.stderr)
        sys.exit(2)

    try:
        video_id, clean_url = base.normalize_youtube_url(url)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    def env_float(name, default):
        try:
            return float(os.environ.get(name, default))
        except ValueError:
            return float(default)

    max_minutes = env_float("YOUTUBE_MAX_MINUTES", base.DEFAULT_MAX_MINUTES)
    max_mb = env_float("YOUTUBE_VIDEO_MAX_MB", DEFAULT_MAX_MB)

    cookie_path = base.write_cookie_file(base.load_cookie_text())
    path = actual = None
    try:
        for attempt in range(1, base.ATTEMPTS + 1):
            print(f"[attempt {attempt}/{base.ATTEMPTS}] Downloading video {video_id} (up to {height}p)...")
            try:
                path, actual = download_video(clean_url, stem, cookie_path, max_minutes, height, max_mb)
                break
            except base.UserFacingError as e:
                message, retryable = str(e), e.retryable
            except Exception as e:
                message, retryable = base.classify_error(str(e), bool(cookie_path))
            if not retryable or attempt == base.ATTEMPTS:
                print(f"ERROR: {message}", file=sys.stderr)
                sys.exit(1)
            print(f"  failed ({message}); retrying in {base.RETRY_PAUSE_S * attempt}s...")
            time.sleep(base.RETRY_PAUSE_S * attempt)
    finally:
        if cookie_path:
            try:
                os.remove(cookie_path)
            except OSError:
                pass

    size_mb = os.path.getsize(path) / 1_048_576
    shown = f", {actual:.0f}s" if actual else ""
    print(f"Downloaded '{path}' ({size_mb:.1f} MB{shown}).")
    with open(result_file, "w", encoding="utf-8") as f:
        f.write(os.path.basename(path))


if __name__ == "__main__":
    main()
