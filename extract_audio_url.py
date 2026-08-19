"""
Opens a NotebookLM Audio Overview share page in a headless browser and
captures the real, signed direct audio URL (googlevideo.com) by watching
network traffic. This must run on the SAME machine/IP that will later
download the file, since the captured URL is IP-locked to the requester.

Usage:
    python extract_audio_url.py <share_url> <output_txt_path>
"""

import re
import sys
import time

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

AUDIO_URL_PATTERN = re.compile(r"googlevideo\.com/videoplayback.*[?&]mime=audio", re.IGNORECASE)

NAV_TIMEOUT_MS = 60_000
CAPTURE_TIMEOUT_S = 45
PLAY_BUTTON_SELECTORS = [
    'button[aria-label*="Play" i]',
    'button[aria-label*="listen" i]',
    '[data-testid*="play" i]',
    'button:has-text("Play")',
]


def try_click_play(page) -> bool:
    for selector in PLAY_BUTTON_SELECTORS:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0:
                locator.click(timeout=5_000)
                print(f"Clicked play button via selector: {selector}")
                return True
        except Exception:
            continue
    return False


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python extract_audio_url.py <share_url> <output_txt_path>", file=sys.stderr)
        sys.exit(1)

    share_url = sys.argv[1]
    output_path = sys.argv[2]

    captured_url = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        def handle_response(response):
            nonlocal captured_url
            if captured_url is None and AUDIO_URL_PATTERN.search(response.url):
                captured_url = response.url
                print("Captured direct audio URL from network traffic.")

        page.on("response", handle_response)

        print(f"Opening share link: {share_url}")
        try:
            page.goto(share_url, wait_until="networkidle", timeout=NAV_TIMEOUT_MS)
        except PlaywrightTimeoutError:
            print("Warning: page did not reach networkidle in time, continuing anyway.")

        if captured_url is None:
            print("No audio request seen yet, attempting to trigger playback...")
            try_click_play(page)

        waited = 0
        while captured_url is None and waited < CAPTURE_TIMEOUT_S:
            page.wait_for_timeout(1_000)
            waited += 1

        browser.close()

    if not captured_url:
        print(
            "ERROR: could not capture a direct audio URL from the share page "
            "within the timeout. The page structure may have changed, or "
            "playback did not start automatically.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(captured_url.strip())

    print(f"Saved direct audio URL to '{output_path}'.")


if __name__ == "__main__":
    main()
