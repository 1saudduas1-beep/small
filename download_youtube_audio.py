"""
Downloads the best audio track of ONE YouTube video with yt-dlp.

Usage:
    python download_youtube_audio.py <youtube_url> <output_stem> <result_file>

    The audio is saved as <output_stem>.<ext> (ext chosen by YouTube, usually
    m4a/webm) and the final file name is written to <result_file> so the
    workflow can pass it to transcribe.py.

Requirements on the runner (done by the workflow):
    * pip install -U "yt-dlp[default]"   (includes the yt-dlp-ejs solver scripts)
    * a JavaScript runtime on PATH (Deno), required by yt-dlp for YouTube
    * ffprobe (ffmpeg package) for the post-download integrity check

Optional env vars:
    YOUTUBE_COOKIES        the raw text of a Netscape cookies.txt (use a
                           throwaway account); pasting it straight into the
                           GitHub secret works, and lines whose tabs were
                           turned into spaces are repaired automatically.
    YOUTUBE_COOKIES_B64    alternative: the same file base64-encoded.
                           Cookies are needed on most cloud runners, where
                           YouTube often answers "Sign in to confirm you're
                           not a bot". They are written to a private temp
                           file and deleted afterwards; never printed.
    YOUTUBE_MAX_MINUTES    refuse videos longer than this (default 240)

Only single-video links are accepted (youtube.com / youtu.be); playlists,
channels and live streams are rejected with a clear message.
"""

import base64
import binascii
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import parse_qs, urlparse

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be"}
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
DEFAULT_MAX_MINUTES = 240
ATTEMPTS = 3
RETRY_PAUSE_S = 10


class UserFacingError(Exception):
    """A problem with a clear, actionable message."""

    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


# ---------------------------------------------------------------- helpers
def normalize_youtube_url(url: str):
    """Return (video_id, canonical_watch_url) or raise ValueError."""
    p = urlparse(url.strip())
    host = (p.hostname or "").lower()
    if p.scheme != "https" or host not in YOUTUBE_HOSTS or p.username or p.password or p.port:
        raise ValueError("the link must be an https youtube.com / youtu.be video link")

    vid = None
    if host == "youtu.be":
        vid = p.path.strip("/").split("/")[0]
    else:
        parts = [x for x in p.path.split("/") if x]
        first = parts[0] if parts else ""
        if first == "watch":
            vid = (parse_qs(p.query).get("v") or [""])[0]
        elif first in ("shorts", "embed", "live", "v") and len(parts) >= 2:
            vid = parts[1]
        elif first in ("playlist", "channel", "c", "user") or first.startswith("@"):
            raise ValueError("playlist/channel links are not supported; use a single video link")

    if not vid or not VIDEO_ID_RE.match(vid):
        raise ValueError("could not find a valid 11-character video id in the link")
    return vid, f"https://www.youtube.com/watch?v={vid}"


def normalize_cookie_text(text: str):
    """Return (clean Netscape text, number of cookies) or (None, 0) if unusable."""
    text = (text or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if text.lstrip().startswith(("[", "{")):
        print("WARNING: the cookies look like JSON; export them in Netscape (cookies.txt) format.")
        return None, 0

    lines, count = [], 0
    for raw in text.split("\n"):
        line = raw.strip("\n ")
        if not line.strip():
            continue
        if line.startswith("#") and not line.startswith("#HttpOnly_"):
            continue  # comments / header: regenerated below
        fields = line.split("\t")
        if len(fields) < 6:
            fields = line.split(None, 6)  # tabs were turned into spaces by copy/paste
        if len(fields) == 6:
            fields.append("")  # cookie with an empty value
        if len(fields) != 7:
            continue
        domain, subdomains, path, secure, expires, name, value = fields
        if subdomains.upper() not in ("TRUE", "FALSE") or secure.upper() not in ("TRUE", "FALSE"):
            continue
        if not re.fullmatch(r"-?\d+", expires.strip()):
            continue
        lines.append("\t".join([domain, subdomains.upper(), path, secure.upper(), expires.strip(), name, value]))
        count += 1

    joined = "\n".join(lines)
    if not count or ("youtube.com" not in joined and "google.com" not in joined):
        return None, 0
    return "# Netscape HTTP Cookie File\n" + joined + "\n", count


def load_cookie_text():
    """Raw YOUTUBE_COOKIES first, else base64 YOUTUBE_COOKIES_B64; None when absent."""
    raw = os.environ.get("YOUTUBE_COOKIES", "")
    if raw.strip():
        return raw
    compact = "".join(os.environ.get("YOUTUBE_COOKIES_B64", "").split())
    if not compact:
        return None
    try:
        return base64.b64decode(compact, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        print("WARNING: YOUTUBE_COOKIES_B64 is not valid base64 text; continuing without cookies.")
        return None


def write_cookie_file(text):
    """Write the cookies to a private temp file; returns its path (or None)."""
    if not text:
        return None
    clean, count = normalize_cookie_text(text)
    if not clean:
        print("WARNING: the cookies secret has no usable youtube.com/google.com cookies; ignoring it.")
        return None
    fd, path = tempfile.mkstemp(prefix="yt_cookies_", suffix=".txt")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(clean)
    os.chmod(path, 0o600)
    print(f"Using {count} provided YouTube/Google cookies.")
    return path


def classify_error(message: str, has_cookies: bool):
    """Map a yt-dlp error to (actionable message, retryable)."""
    m = message.lower()

    if "confirm your age" in m or "age-restricted" in m or "inappropriate for some users" in m:
        return ("This video is age-restricted: it needs cookies from an adult YouTube account "
                "(secret YOUTUBE_COOKIES).", False)
    if "not a bot" in m:
        if has_cookies:
            return ("YouTube flagged the runner even with cookies: the cookies are probably expired or "
                    "rotated. Re-export them from a fresh private/incognito window (log in, open "
                    "youtube.com/robots.txt, export, close the window) and update the secret. "
                    "A flagged datacenter IP can also be the cause; retry later.", False)
        return ("YouTube blocked this runner (\"Sign in to confirm you're not a bot\"). Add the secret "
                "YOUTUBE_COOKIES (paste the cookies.txt of a throwaway account).", False)
    if "private video" in m:
        return ("This video is private.", False)
    if "members-only" in m or "join this channel" in m:
        return ("This is a members-only video.", False)
    if "not available in your country" in m or "blocked it in your country" in m or "blocked it on copyright" in m:
        return ("This video is blocked in the runner's country / for copyright reasons.", False)
    if "premieres in" in m or "live event will begin" in m:
        return ("This is an upcoming live event/premiere; try again after it ends.", False)
    if "javascript runtime" in m or "n challenge" in m or "requested format is not available" in m:
        return ("yt-dlp could not solve YouTube's JavaScript challenge. Make sure Deno is installed "
                "and yt-dlp is the latest version installed as \"yt-dlp[default]\".", False)
    if "video unavailable" in m or "this video is not available" in m or "has been removed" in m or "has been terminated" in m:
        return ("The video is unavailable (removed, region-limited, or the JavaScript challenge solver "
                "is missing).", False)
    if "http error 403" in m:
        return ("YouTube answered 403 Forbidden (IP flagged or expired session).", True)
    return (message.strip().splitlines()[-1][:300] if message.strip() else "unknown yt-dlp error", True)


def probe_duration(path: str):
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=120,
        )
        return float(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError, OSError):
        return None


def pick_output(result, stem: str):
    """Locate the downloaded file."""
    if isinstance(result, dict):
        for item in result.get("requested_downloads") or []:
            path = item.get("filepath")
            if path and os.path.isfile(path):
                return path
    candidates = [p for p in glob.glob(glob.escape(stem) + ".*")
                  if not p.endswith((".part", ".ytdl", ".temp"))]
    candidates = [p for p in candidates if os.path.getsize(p) > 0]
    return max(candidates, key=os.path.getmtime) if candidates else None


# ---------------------------------------------------------------- download
def download(url: str, stem: str, cookie_path, max_minutes: float):
    import yt_dlp  # imported here so the helpers above are usable without it

    if not shutil.which("deno"):
        print("WARNING: no 'deno' on PATH; YouTube extraction may fail without a JavaScript runtime.")

    for stale in glob.glob(glob.escape(stem) + ".*"):
        try:
            os.remove(stale)
        except OSError:
            pass

    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": {"default": f"{stem}.%(ext)s"},
        "noplaylist": True,
        "restrictfilenames": True,
        "retries": 5,
        "fragment_retries": 5,
        "extractor_retries": 3,
        "socket_timeout": 30,
        "noprogress": True,
    }
    if cookie_path:
        opts["cookiefile"] = cookie_path

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

        if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
            raise UserFacingError("Live streams / upcoming premieres are not supported.")
        duration = info.get("duration")
        if duration and duration > max_minutes * 60:
            raise UserFacingError(
                f"The video is {duration / 60:.0f} min long; the limit is {max_minutes:.0f} min "
                "(raise YOUTUBE_MAX_MINUTES if you really need it)."
            )
        title = re.sub(r"\s+", " ", str(info.get("title") or "")).strip()[:120]
        print(f"Video: {title} | duration: {(duration or 0) / 60:.1f} min")

        try:
            result = ydl.process_ie_result(info, download=True)
        except yt_dlp.utils.DownloadError:
            raise
        except (AttributeError, TypeError, KeyError):
            ydl.download([url])  # extra safety net: let yt-dlp redo the extraction itself
            result = None

    path = pick_output(result, stem)
    if not path:
        raise UserFacingError("yt-dlp finished but no audio file was produced.", retryable=True)

    actual = probe_duration(path)
    if duration and actual is not None and abs(actual - duration) > max(5.0, 0.02 * duration):
        os.remove(path)
        raise UserFacingError(
            f"Downloaded audio is {actual:.0f}s but the video is {duration:.0f}s (partial download).",
            retryable=True,
        )
    return path, actual


def main():
    if len(sys.argv) != 4:
        print("Usage: python download_youtube_audio.py <youtube_url> <output_stem> <result_file>", file=sys.stderr)
        sys.exit(1)

    url, stem, result_file = sys.argv[1], sys.argv[2], sys.argv[3]

    try:
        video_id, clean_url = normalize_youtube_url(url)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        max_minutes = float(os.environ.get("YOUTUBE_MAX_MINUTES", DEFAULT_MAX_MINUTES))
    except ValueError:
        max_minutes = DEFAULT_MAX_MINUTES

    cookie_path = write_cookie_file(load_cookie_text())
    path = None
    try:
        for attempt in range(1, ATTEMPTS + 1):
            print(f"[attempt {attempt}/{ATTEMPTS}] Downloading audio for video {video_id}...")
            try:
                path, actual = download(clean_url, stem, cookie_path, max_minutes)
                break
            except UserFacingError as e:
                message, retryable = str(e), e.retryable
            except Exception as e:  # yt-dlp errors, network problems, ...
                message, retryable = classify_error(str(e), bool(cookie_path))

            if not retryable or attempt == ATTEMPTS:
                print(f"ERROR: {message}", file=sys.stderr)
                sys.exit(1)
            print(f"  failed ({message}); retrying in {RETRY_PAUSE_S * attempt}s...")
            time.sleep(RETRY_PAUSE_S * attempt)
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
