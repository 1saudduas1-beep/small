"""
Opens a NotebookLM Audio Overview share page in a headless browser and
captures the real, signed direct audio URL (googlevideo.com) by watching
network traffic. This must run on the SAME machine/IP that will later
download the file, since the captured URL is IP-locked to the requester.

Hardening / reliability:
  * Only https links on Google domains (*.google.com, *.google) are opened.
  * No `networkidle` wait (it can never settle on pages that poll/stream);
    the script waits for the audio request itself and re-clicks Play.
  * Up to ATTEMPTS fresh browser sessions before giving up.
  * A `range=` query parameter (a partial-content window) is stripped so the
    full file is downloaded; the URL itself is never printed to the log.
  * On failure a screenshot is saved to debug_extract.png for diagnosis.

Usage:
    python extract_audio_url.py <share_url> <output_txt_path>
"""

import re
import sys
import time
from urllib.parse import urlparse

AUDIO_URL_PATTERN = re.compile(r"googlevideo\.com/videoplayback.*[?&]mime=audio", re.IGNORECASE)
HOST_PATTERN = re.compile(r"^([a-z0-9-]+\.)+google(\.com)?$", re.IGNORECASE)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

ATTEMPTS = 3
NAV_TIMEOUT_MS = 60_000
CAPTURE_TIMEOUT_S = 60
CLICK_INTERVAL_S = 6
RETRY_PAUSE_S = 5
SCREENSHOT_PATH = "debug_extract.png"

PLAY_BUTTON_SELECTORS = [
    'button[aria-label*="Play" i]',
    'button[aria-label*="listen" i]',
    '[data-testid*="play" i]',
    'button:has-text("Play")',
]


class NeedsSignIn(Exception):
    """The share link redirected to a Google sign-in page (notebook not public)."""


def is_allowed_share_url(url: str) -> bool:
    """https only, on a Google domain, without credentials or a custom port."""
    try:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port:
            return False
        return bool(parsed.hostname and HOST_PATTERN.match(parsed.hostname))
    except ValueError:
        return False


def strip_range_param(url: str) -> str:
    """Drop a `range=` query parameter without re-encoding the signed ones."""
    if "?" not in url:
        return url
    base, query = url.split("?", 1)
    kept = [p for p in query.split("&") if not p.lower().startswith("range=")]
    return base + "?" + "&".join(kept)


def try_click_play(page) -> bool:
    for selector in PLAY_BUTTON_SELECTORS:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0:
                locator.click(timeout=3_000)
                print(f"Clicked play button via selector: {selector}")
                return True
        except Exception:
            continue
    return False


def run_attempt(p, share_url: str, attempt: int):
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    browser = p.chromium.launch(
        headless=True,
        args=["--autoplay-policy=no-user-gesture-required"],
    )
    try:
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 900},
            locale="en-US",
        )
        page = context.new_page()
        found = []

        def on_url(url: str) -> None:
            if not found and AUDIO_URL_PATTERN.search(url):
                found.append(url)
                print("Captured direct audio URL from network traffic.")

        page.on("request", lambda r: on_url(r.url))
        page.on("response", lambda r: on_url(r.url))

        print(f"[attempt {attempt}/{ATTEMPTS}] Opening share link...")
        try:
            page.goto(share_url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            print("Warning: page did not finish loading in time, continuing anyway.")

        final_host = urlparse(page.url).hostname or ""
        if final_host == "accounts.google.com":
            raise NeedsSignIn()

        deadline = time.time() + CAPTURE_TIMEOUT_S
        next_click = 0.0
        while not found and time.time() < deadline:
            if time.time() >= next_click:
                try_click_play(page)
                next_click = time.time() + CLICK_INTERVAL_S
            page.wait_for_timeout(500)

        if not found:
            try:
                page.screenshot(path=SCREENSHOT_PATH, full_page=True)
                print(f"Saved diagnostic screenshot to {SCREENSHOT_PATH}.")
            except Exception:
                pass
        return found[0] if found else None
    finally:
        browser.close()


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python extract_audio_url.py <share_url> <output_txt_path>", file=sys.stderr)
        sys.exit(1)

    share_url = sys.argv[1].strip()
    output_path = sys.argv[2]

    if not is_allowed_share_url(share_url):
        print(
            "ERROR: share_url must be an https link on a Google domain "
            "(*.google.com or *.google).",
            file=sys.stderr,
        )
        sys.exit(2)

    from playwright.sync_api import sync_playwright

    captured_url = None
    with sync_playwright() as p:
        for attempt in range(1, ATTEMPTS + 1):
            try:
                captured_url = run_attempt(p, share_url, attempt)
            except NeedsSignIn:
                print(
                    "ERROR: the link redirected to a Google sign-in page. Make the notebook "
                    "public ('Anyone with a link', full notebook access) and share again.",
                    file=sys.stderr,
                )
                sys.exit(1)
            except Exception as e:  # browser crash, navigation error, ...
                print(f"Attempt {attempt} failed: {e!r}")
            if captured_url:
                break
            if attempt < ATTEMPTS:
                time.sleep(RETRY_PAUSE_S)

    if not captured_url:
        print(
            "ERROR: could not capture a direct audio URL from the share page "
            f"after {ATTEMPTS} attempts. The page structure may have changed, "
            "or playback did not start (see debug_extract.png).",
            file=sys.stderr,
        )
        sys.exit(1)

    cleaned = strip_range_param(captured_url.strip())
    if cleaned != captured_url.strip():
        print("Removed 'range' parameter from the captured URL (full-file download).")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Saved direct audio URL to '{output_path}'.")


if __name__ == "__main__":
    main()
