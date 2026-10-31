"""
Narrates the Arabic translation (<name>_ar.txt or <name>_ar.srt produced by translate_transcript.py)
as ONE audio file (MP3, optional WAV) with Gemini 3.8 Flash-Lite TTS, single narrator, free tier.

Usage:
    python narrate_arabic.py <input_file> <output_prefix> [--plan]

    --plan   prints the chunk plan, request count and number of days needed; makes no API call.

Pipeline
  1. READ     .txt (paragraphs separated by blank lines) or .srt (cues merged into paragraphs).
              Lines without Arabic letters (English lines of bilingual files) are dropped.
  2. CLEAN    for speech: "English (عربي)" -> "English، أي عربي،" (GLOSS: ay | comma | skip), attached
              prefixes before Latin terms (بـx-ray) get a vowel mark, parentheses become pauses,
              tatweel / markup / quotes removed, % spoken, every paragraph ends with punctuation.
              English terms stay in Latin letters so the narrator pronounces them in English.
  3. CHUNK    paragraph/sentence boundaries, <= NARRATE_CHUNK_CHARS characters per request.
  4. SYNTH    one request per chunk; one worker per API key runs in parallel (each key respects its own RPM),
              a per-day 429 retires that key and its chunk goes back to the queue. Every finished chunk is saved to disk at once (resume = run again).
              Truncated / too-short audio is detected and the chunk is split in two and redone.
  5. ASSEMBLE chunks + pauses -> WAV -> MP3 (loudness-normalised) + chapters + report + spoken script.
              If the quota ends early, the finished leading part is written as *_partial* and exit code is 2.

Environment
    GEMINI_API_KEY, GEMINI_API_KEY_2 ... _8, GEMINI_API_KEYS   (create each key in a DIFFERENT Google
                          Cloud project: the free quota is per project)
    NARRATE_MODEL         default gemini-3.8-flash-lite-tts
    NARRATE_VOICE         default Charon (a prebuilt voice name, or a designed voice id voice_...)
    NARRATE_STYLE         narration style sent as speech_metadata.style (never spoken); empty = none
    NARRATE_CHUNK_CHARS   default 4500 (clamped 800-7000)
    NARRATE_GLOSS         ay (default) | comma | skip
    NARRATE_RPM           default 3 (free tier) ; NARRATE_RPM_MARGIN seconds added to the gap (default 2.5)
    NARRATE_MAX_MINUTES   time budget (default 120)
    NARRATE_WAV           off (default) | on   (also write the full WAV)
    NARRATE_MP3_KBPS      default 96
    NARRATE_WORKDIR       default narration_work
    NARRATE_WORKERS       parallel workers, one key each (default 0 = one per key; 1 = sequential)
    GEMINI_API_BASE       override the API base (tests)

Exit codes: 0 complete | 2 incomplete (quota / time / chunk failure; partial output written) | 1 fatal.
Uses only the Python standard library (+ ffmpeg for MP3 and rare resampling).
"""

import base64
import hashlib
import io
import json
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from collections import Counter

API_BASE = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
DEFAULT_MODEL = "gemini-3.8-flash-lite-tts"
DEFAULT_VOICE = "Charon"
DEFAULT_STYLE = ("calm, clear and authoritative educational narration in Modern Standard Arabic, steady natural "
                 "pace; English medical and scientific terms are pronounced in clear English")

RATE = 24000                 # standard PCM rate of the assembled audio (Gemini TTS output is 24 kHz mono)
CPS = 12.5                   # expected narrated characters per second (Arabic, calm pace)
SHORT_RATIO = 0.55           # audio shorter than this share of the expected duration = text was skipped
PAUSE_CHUNK = 0.55           # seconds of silence between chunks
PAUSE_START, PAUSE_END = 0.4, 0.8
MAX_ATTEMPTS = 3             # transient failures per chunk (each request may count against the daily quota)
HTTP_TIMEOUT = 420

AR_RE = re.compile(r"[\u0600-\u06FF]")
TS_RE = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->")
SENT_END = re.compile(r"[.?!\u061f\u06d4\u2026][\"')\]\u00bb\u201d]*$")
LATIN_RUN = r"[A-Za-z][A-Za-z0-9'\u2019.+/-]*"
GLOSS_RE = re.compile(r"(" + LATIN_RUN + r"(?:\s+" + LATIN_RUN + r")*)\s*\(\s*([^()A-Za-z]*[\u0600-\u06FF][^()A-Za-z]*?)\s*\)")
PREFIX_MARK = {"ب": "بِ", "ل": "لِ", "ف": "فَ", "ك": "كَ"}


class QuotaExhausted(Exception):
    pass


class TimeUp(Exception):
    pass


class Fatal(Exception):
    pass


class ChunkFailed(Exception):
    pass


class BadResponse(Exception):
    pass


_LOG_LOCK = threading.Lock()


def log(msg):
    name = threading.current_thread().name
    prefix = f"[{name}] " if name.startswith("key") else ""
    with _LOG_LOCK:
        print(prefix + msg, flush=True)


def env_num(name, default, cast=float):
    try:
        return cast(os.environ.get(name, "").strip() or default)
    except ValueError:
        return cast(default)


class Cfg:
    def __init__(self):
        self.model = os.environ.get("NARRATE_MODEL", "").strip() or DEFAULT_MODEL
        self.voice = os.environ.get("NARRATE_VOICE", "").strip() or DEFAULT_VOICE
        self.style = os.environ["NARRATE_STYLE"].strip() if "NARRATE_STYLE" in os.environ else DEFAULT_STYLE
        self.chunk = max(800, min(7000, env_num("NARRATE_CHUNK_CHARS", 4500, int)))
        g = os.environ.get("NARRATE_GLOSS", "ay").strip().lower()
        self.gloss = g if g in ("ay", "comma", "skip") else "ay"
        self.rpm = max(0.5, env_num("NARRATE_RPM", 3.0))
        self.gap = 60.0 / self.rpm + max(0.0, env_num("NARRATE_RPM_MARGIN", 2.5))
        self.minutes = max(5.0, env_num("NARRATE_MAX_MINUTES", 120.0))
        self.wav = os.environ.get("NARRATE_WAV", "off").strip().lower() in ("on", "1", "true", "yes")
        self.kbps = max(32, env_num("NARRATE_MP3_KBPS", 96, int))
        self.workdir = os.environ.get("NARRATE_WORKDIR", "narration_work").strip() or "narration_work"
        self.workers = max(0, env_num("NARRATE_WORKERS", 0, int))   # 0 = one worker per key


def load_keys():
    names = ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{i}" for i in range(2, 9)]
    vals = [os.environ.get(n, "").strip() for n in names]
    vals += [v.strip() for v in os.environ.get("GEMINI_API_KEYS", "").split(",")]
    out = []
    for v in vals:
        if v and v not in out:
            out.append(v)
    return out


# ------------------------------------------------------------------ reading
def read_paragraphs(path):
    """Returns (paragraphs, is_srt)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read().replace("\r\n", "\n")
    items, is_srt = [], False
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if lines and lines[0].isdigit():
            lines = lines[1:]
        if lines and TS_RE.match(lines[0]):
            is_srt = True
            lines = lines[1:]
        lines = [re.sub(r"^[-\u2013\u2014]+\s*", "", l) for l in lines if AR_RE.search(l)]
        text = " ".join(l for l in lines if l).strip()
        if text:
            items.append(text)
    if not is_srt:
        return items, False
    paras, cur = [], ""
    for t in items:
        cur = (cur + " " + t).strip()
        if (SENT_END.search(cur) and len(cur) >= 300) or len(cur) >= 900:
            paras.append(cur)
            cur = ""
    if cur:
        paras.append(cur)
    return paras, True


# ------------------------------------------------------------------ cleaning
def clean_for_speech(text, gloss="ay"):
    t = text.replace("\u27e6", "").replace("\u27e7", "")

    def gl(m):
        term, ar = m.group(1).strip(), re.sub(r"\s+", " ", m.group(2)).strip()
        if gloss == "skip":
            return term
        return f"{term}\u060c \u0623\u064a {ar}\u060c" if gloss == "ay" else f"{term}\u060c {ar}\u060c"

    t = GLOSS_RE.sub(gl, t)
    t = re.sub(r"(?<![\u0600-\u06FF])([\u0628\u0644\u0641\u0643])\u0640+(?=[A-Za-z])",
               lambda m: PREFIX_MARK[m.group(1)] + " ", t)
    t = re.sub(r"\u0640+(?=[A-Za-z])", " ", t).replace("\u0640", "")
    t = re.sub(r"[(\[{]\s*([^()\[\]{}]*?)\s*[)\]}]", "\u060c \\1\u060c", t)
    t = re.sub(r"[\"\u201c\u201d\u00ab\u00bb]", "", t)
    t = re.sub(r"\s[-\u2013\u2014]+\s", "\u060c ", t)
    t = t.replace("\u2026", ".").replace("|", " ")
    t = re.sub(r"(\d)\s*%", "\\1 \u0628\u0627\u0644\u0645\u0626\u0629", t)
    t = re.sub(r"[\u0000-\u0008\u000b-\u001f\U0001F000-\U0001FAFF\u2600-\u27BF]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([\u060c.\u061f!\u061b:])", r"\1", t)
    t = re.sub(r"\u060c\s*([.\u061f!\u061b:])", r"\1", t)
    t = re.sub(r"([\u060c\u061b])(\s*\1)+", r"\1", t)
    t = re.sub(r"^[\u060c\s]+", "", t)
    if t and not re.search(r"[.\u061f!:\u061b\u2026]$", t):
        t += "."
    return t


# ------------------------------------------------------------------ chunking
def split_long(text, limit):
    """A paragraph longer than `limit` -> pieces at sentence ends (then commas / spaces)."""
    if len(text) <= limit:
        return [text]
    out, cur = [], ""
    for s in re.split(r"(?<=[.\u061f!\u2026])\s+", text):
        while len(s) > limit:
            cut = max(s.rfind("\u060c ", 0, limit), s.rfind(", ", 0, limit))
            if cut < limit // 2:
                cut = s.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit - 1
            piece, s = s[:cut + 1].strip(), s[cut + 1:].strip()
            if cur:
                out.append(cur)
                cur = ""
            out.append(piece)
        if cur and len(cur) + 1 + len(s) > limit:
            out.append(cur)
            cur = ""
        cur = (cur + " " + s).strip()
    if cur:
        out.append(cur)
    return [x for x in out if x]


def build_chunks(paras, limit):
    chunks, cur = [], ""
    for p in paras:
        for piece in split_long(p, limit):
            if cur and len(cur) + 2 + len(piece) > limit:
                chunks.append(cur)
                cur = ""
            cur = (cur + "\n\n" + piece) if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def halve(text):
    cands = [m.end() for m in re.finditer(r"\n\n|(?<=[.\u061f!\u2026])\s+", text)]
    if not cands:
        return text[:len(text) // 2], text[len(text) // 2:]
    cut = min(cands, key=lambda c: abs(c - len(text) / 2))
    return text[:cut].strip(), text[cut:].strip()


def chunk_key(text, cfg):
    h = hashlib.sha1(f"{cfg.model}|{cfg.voice}|{cfg.style}|{text}".encode("utf-8")).hexdigest()
    return h[:16]


# ------------------------------------------------------------------ audio
def parse_wav(data):
    """WAV bytes -> (pcm s16le, rate, channels). Tolerates streaming headers with an unknown data size."""
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = pos + 8
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", data[body:body + 16])
        elif cid == b"data":
            end = body + size
            if size == 0 or size == 0xFFFFFFFF or end > len(data):
                end = len(data)
            pcm = data[body:end]
            break
        pos = body + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError("WAV without fmt/data chunk")
    if fmt[0] not in (1, 0xFFFE) or fmt[5] != 16:
        raise ValueError(f"unsupported WAV format (tag={fmt[0]}, bits={fmt[5]})")
    return pcm[:len(pcm) - (len(pcm) % (2 * fmt[1]))], fmt[2], fmt[1]


def decode_audio(blob, mime):
    if blob[:4] == b"RIFF":
        return parse_wav(blob)
    m = re.search(r"rate=(\d+)", mime or "")
    return blob[:len(blob) - (len(blob) % 2)], int(m.group(1)) if m else RATE, 1


def to_standard(pcm, rate, ch):
    """Mono, 24 kHz, s16le."""
    if rate == RATE and ch == 1:
        return pcm
    if not shutil.which("ffmpeg"):
        raise BadResponse(f"audio is {rate} Hz / {ch} ch and ffmpeg is not available to convert it")
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(rate), "-ac", str(ch), "-i", "-",
                        "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-"], input=pcm, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise BadResponse("ffmpeg could not convert the audio")
    return r.stdout


def make_wav(pcm):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def silence(sec):
    return b"\x00\x00" * int(RATE * sec)


def dur(pcm):
    return len(pcm) / (2.0 * RATE)


def stamp(sec):
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ------------------------------------------------------------------ API
def extract_audio(out):
    cands = out.get("candidates") or []
    if not cands:
        raise BadResponse(f"no candidates (blockReason={(out.get('promptFeedback') or {}).get('blockReason')})")
    c = cands[0]
    finish = c.get("finishReason")
    pcms = []
    for p in (c.get("content") or {}).get("parts") or []:
        d = p.get("inlineData") or p.get("inline_data")
        if d and d.get("data"):
            pcm, rate, ch = decode_audio(base64.b64decode(d["data"]), d.get("mimeType") or d.get("mime_type") or "")
            pcms.append(to_standard(pcm, rate, ch))
    if not pcms:
        raise BadResponse(f"no audio in the response (finishReason={finish})")
    return b"".join(pcms), finish


def post_json(url, key, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class Api:
    def __init__(self, keys, cfg):
        self.keys, self.cfg = keys, cfg
        self.bad, self.exhausted = set(), set()
        self.next_ok = {}
        self.used = Counter()
        self.errs = Counter()
        self.quirks = []
        self.waited = 0.0
        self.deadline = time.time() + cfg.minutes * 60
        self.voice_form = 1 if cfg.voice.startswith("voice_") else 0
        self.send_style = bool(cfg.style)

    def live(self):
        return [i for i in range(len(self.keys)) if i not in self.bad and i not in self.exhausted]

    def check(self):
        if time.time() > self.deadline:
            raise TimeUp("time budget reached")

    def sleep(self, s):
        end = time.time() + s
        while True:
            self.check()
            left = end - time.time()
            if left <= 0:
                return
            step = min(left, 1.0)
            time.sleep(step)
            self.waited += step

    def pick(self):
        live = self.live()
        if not live:
            raise QuotaExhausted("every API key is rejected or has used its daily quota")
        ki = min(live, key=lambda k: self.next_ok.get(k, 0.0))
        wait = self.next_ok.get(ki, 0.0) - time.time()
        if wait > 0:
            self.sleep(wait)
        return ki

    def body(self, text):
        part = {"text": text}
        if self.send_style:
            part["speech_metadata"] = {"style": self.cfg.style}
        v = self.cfg.voice
        voice = {"prebuiltVoiceConfig": {"voiceName": v}} if self.voice_form == 0 else {"voice": v}
        return {"contents": [{"role": "user", "parts": [part]}],
                "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {"voiceConfig": voice}}}

    def degrade(self, raw):
        low = raw.lower()
        if self.send_style and ("speech_metadata" in low or "speechmetadata" in low or "style" in low):
            self.send_style = False
            self.quirks.append("speech_metadata rejected (HTTP 400) -> narration style no longer sent")
            return True
        if self.voice_form == 0 and ("prebuilt" in low or "voice" in low):
            self.voice_form = 1
            self.quirks.append("prebuiltVoiceConfig rejected (HTTP 400) -> using voiceConfig.voice")
            return True
        return False

    def synthesize(self, text):
        """-> (pcm, finishReason). Rotates keys, honours RPM, handles quota errors."""
        url = f"{API_BASE}/models/{self.cfg.model}:generateContent"
        attempts = degrades = limited = 0
        last = None
        while True:
            self.check()
            if attempts >= MAX_ATTEMPTS:
                raise ChunkFailed(last or "failed")
            ki = self.pick()
            self.next_ok[ki] = time.time() + self.cfg.gap
            self.used[ki] += 1
            try:
                out = post_json(url, self.keys[ki], self.body(text), HTTP_TIMEOUT)
                return extract_audio(out)
            except urllib.error.HTTPError as e:
                raw = e.read().decode("utf-8", "ignore") if hasattr(e, "read") else ""
                self.errs[e.code] += 1
                last = f"HTTP {e.code}: {raw[:200]}"
                if e.code == 400:
                    if degrades < 3 and self.degrade(raw):
                        degrades += 1
                        log(f"  {self.quirks[-1]}")
                        continue
                    raise Fatal(f"HTTP 400 from the API: {raw[:400]}")
                if e.code in (401, 403):
                    self.bad.add(ki)
                    log(f"  key #{ki + 1} rejected (HTTP {e.code}) -> not used again")
                    continue
                if e.code == 404:
                    raise Fatal(f"model {self.cfg.model} not found (HTTP 404). Check NARRATE_MODEL.")
                if e.code == 429:
                    low = raw.lower()
                    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', raw) or re.search(r"retry in (\d+(?:\.\d+)?)s", raw)
                    delay = float(m.group(1)) if m else 30.0
                    if "perday" in low or "per day" in low or "limit: 0" in low or '"limit": 0' in low or delay > 300:
                        self.exhausted.add(ki)
                        log(f"  key #{ki + 1}: daily quota used up")
                        continue
                    limited += 1
                    if limited > 40:
                        raise ChunkFailed("rate limited too many times")
                    self.next_ok[ki] = time.time() + min(delay, 120) + random.uniform(1, 3)
                    log(f"  key #{ki + 1}: per-minute limit, waiting {min(delay, 120):.0f}s")
                    continue
                attempts += 1
                log(f"  HTTP {e.code}, retry {attempts}/{MAX_ATTEMPTS}")
                self.sleep(6 * attempts + random.uniform(0, 4))
            except (BadResponse, ValueError) as e:
                attempts += 1
                last = str(e)[:200]
                log(f"  unusable response ({last}), retry {attempts}/{MAX_ATTEMPTS}")
                self.sleep(4 * attempts)
            except (OSError, TimeoutError) as e:
                attempts += 1
                last = repr(e)[:200]
                log(f"  network/timeout ({last}), retry {attempts}/{MAX_ATTEMPTS}")
                self.sleep(6 * attempts)


# ------------------------------------------------------------------ main flow
class Run:
    def __init__(self, cfg, api):
        self.cfg, self.api = cfg, api
        self.work = cfg.workdir
        self.flags, self.splits, self.fresh = [], 0, 0
        os.makedirs(self.work, exist_ok=True)

    def p(self, key, ext):
        return os.path.join(self.work, f"{key}.{ext}")

    def resolve(self, text, depth=0):
        """-> list of chunk keys (audio files) that together narrate `text`."""
        key = chunk_key(text, self.cfg)
        if os.path.isfile(self.p(key, "wav")):
            return [key]
        if os.path.isfile(self.p(key, "json")):
            try:
                with open(self.p(key, "json"), encoding="utf-8") as f:
                    parts = json.load(f).get("parts")
                if parts and all(os.path.isfile(self.p(k, "wav")) for k in parts):
                    return parts
            except (OSError, ValueError):
                pass
        pcm, finish = self.api.synthesize(text)
        self.fresh += 1
        seconds, expected = dur(pcm), len(text) / CPS
        short = seconds < SHORT_RATIO * expected or finish == "MAX_TOKENS"
        if short and depth < 1 and len(text) > 800:
            a, b = halve(text)
            self.splits += 1
            log(f"  audio too short ({seconds:.0f}s for ~{expected:.0f}s expected, finish={finish}) -> splitting the chunk")
            keys = self.resolve(a, depth + 1) + self.resolve(b, depth + 1)
            with open(self.p(key, "json"), "w", encoding="utf-8") as f:
                json.dump({"parts": keys}, f)
            return keys
        with open(self.p(key, "wav"), "wb") as f:
            f.write(make_wav(pcm))
        with open(self.p(key, "json"), "w", encoding="utf-8") as f:
            json.dump({"chars": len(text), "seconds": round(seconds, 1), "finish": finish, "model": self.cfg.model}, f)
        if short:
            self.flags.append(f"{key}: short audio ({seconds:.0f}s, expected ~{expected:.0f}s) - check this part")
        elif seconds > 2.5 * expected:
            self.flags.append(f"{key}: unusually long audio ({seconds:.0f}s, expected ~{expected:.0f}s) - check this part")
        log(f"  saved {seconds:.0f}s of audio ({len(text)} chars)")
        return [key]

    def read_pcm(self, key):
        with open(self.p(key, "wav"), "rb") as f:
            pcm, rate, ch = parse_wav(f.read())
        return to_standard(pcm, rate, ch)


def encode(prefix, wav_path, cfg, partial):
    suffix = "_partial" if partial else ""
    mp3 = f"{prefix}_narration{suffix}.mp3"
    if not shutil.which("ffmpeg"):
        log("WARNING: ffmpeg not found - MP3 skipped, WAV kept.")
        return None
    title = os.path.basename(prefix)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", wav_path, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
           "-ar", str(RATE), "-ac", "1", "-c:a", "libmp3lame", "-b:a", f"{cfg.kbps}k",
           "-metadata", f"title={title}", "-metadata", "genre=Speech", mp3]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"WARNING: MP3 encoding failed: {r.stderr[-300:]}")
        return None
    return mp3


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    plan_only = "--plan" in sys.argv
    if len(args) != 2:
        print("Usage: python narrate_arabic.py <input_file> <output_prefix> [--plan]")
        return 1
    src, prefix = args
    cfg = Cfg()
    if not os.path.isfile(src):
        print(f"ERROR: input '{src}' not found.", file=sys.stderr)
        return 1
    raw_paras, is_srt = read_paragraphs(src)
    paras = [x for x in (clean_for_speech(p, cfg.gloss) for p in raw_paras) if len(re.sub(r"[\W_]+", "", x)) >= 2]
    if not paras:
        print("ERROR: no Arabic text found in the input.", file=sys.stderr)
        return 1
    chunks = build_chunks(paras, cfg.chunk)
    total = sum(len(c) for c in chunks)
    keys = load_keys()
    nkeys = max(1, len(keys))
    log(f"Input: {'srt' if is_srt else 'txt'} | paragraphs: {len(paras)} | chars: {total} | chunks: {len(chunks)} "
        f"(<= {cfg.chunk} chars) | estimated audio: {total / CPS / 60:.0f} min | model: {cfg.model} | voice: {cfg.voice}")
    log(f"Requests needed: {len(chunks)} (+ splits if audio comes back short) | keys: {len(keys)} | "
        f"free tier ~10 requests/day/key -> about {-(-len(chunks) // (10 * nkeys))} day(s); "
        f"minimum {cfg.gap:.0f}s between requests per key")
    if os.path.dirname(prefix):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
    script_path = f"{prefix}_narration_script.txt"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write("\n\n=====\n\n".join(chunks) + "\n")
    if plan_only:
        log(f"Plan only: spoken script written to {script_path}; no API call made.")
        return 0
    if not keys:
        print("ERROR: no Gemini API key found (set GEMINI_API_KEY as a GitHub secret).", file=sys.stderr)
        return 1

    started = time.time()
    n_workers = min(len(keys), cfg.workers or len(keys), 8)
    log(f"Workers: {n_workers} in parallel (one API key each)")
    queue = list(range(len(chunks)))
    results, failures, stops, fatals = {}, {}, [], []
    lock, halt = threading.Lock(), threading.Event()
    workers = []

    def work(ki, api, run):
        while not halt.is_set():
            with lock:
                if not queue:
                    return
                n = queue.pop(0)
            log(f"Chunk {n + 1}/{len(chunks)} ({len(chunks[n])} chars)")
            try:
                ks = run.resolve(chunks[n])
                with lock:
                    results[n] = ks
            except (QuotaExhausted, TimeUp) as e:
                with lock:
                    queue.insert(0, n)
                    queue.sort()
                    stops.append(str(e))
                log(f"worker stopped: {e}")
                return
            except ChunkFailed as e:
                with lock:
                    failures[n] = str(e)
                log(f"chunk {n + 1} failed: {e}")
            except Fatal as e:
                with lock:
                    fatals.append(str(e))
                halt.set()
                return
            except Exception as e:  # unexpected: do not lose the other workers' progress
                with lock:
                    failures[n] = repr(e)[:200]
                log(f"chunk {n + 1} unexpected error: {e!r}")

    threads = []
    for w, ki in enumerate(range(n_workers)):
        api = Api([keys[ki]], cfg)
        run = Run(cfg, api)
        workers.append((ki, api, run))
        th = threading.Thread(target=work, args=(ki, api, run), name=f"key{ki + 1}", daemon=True)
        threads.append(th)
        th.start()
        time.sleep(0.2)
    for th in threads:
        th.join()
    if fatals:
        print(f"ERROR: {fatals[0]}", file=sys.stderr)
        return 1

    class Agg:
        pass

    agg = Agg()
    agg.send_style = all(a.send_style for _, a, _ in workers)
    agg.used, agg.errs = Counter(), Counter()
    agg.bad, agg.exhausted, agg.quirks, agg.flags = set(), set(), [], []
    agg.fresh = agg.splits = 0
    agg.waited = 0.0
    for ki, a, r in workers:
        agg.used[ki] += sum(a.used.values())
        agg.errs.update(a.errs)
        agg.bad |= {ki for _ in a.bad}
        agg.exhausted |= {ki for _ in a.exhausted}
        agg.quirks += [q for q in a.quirks if q not in agg.quirks]
        agg.flags += r.flags
        agg.fresh += r.fresh
        agg.splits += r.splits
        agg.waited += a.waited
    reader = Run(cfg, None)
    agg.read_pcm = reader.read_pcm
    api = run = agg

    done, n = [], 0
    while n in results:          # only the contiguous leading part can be assembled into one file
        done.append((chunks[n], results[n]))
        n += 1
    stop = None
    if len(done) < len(chunks):
        stop = (failures.get(n) and f"chunk {n + 1} failed: {failures[n]}") or (stops[0] if stops else "chunks missing")
        log(f"STOPPED: {len(done)}/{len(chunks)} leading chunks ready ({len(results)} chunks narrated in total): {stop}")

    complete = stop is None
    pcm, chapters, t = [silence(PAUSE_START)], [], PAUSE_START
    for n, (text, ks) in enumerate(done, 1):
        chapters.append((t, n, re.sub(r"\s+", " ", text)[:70]))
        for k in ks:
            seg = run.read_pcm(k) + silence(PAUSE_CHUNK)
            pcm.append(seg)
            t += dur(seg)
    pcm.append(silence(PAUSE_END))
    out_files = []
    if done:
        data = b"".join(pcm)
        suffix = "" if complete else "_partial"
        wav_path = f"{prefix}_narration{suffix}.wav"
        with open(wav_path, "wb") as f:
            f.write(make_wav(data))
        mp3 = encode(prefix, wav_path, cfg, not complete)
        if mp3:
            out_files.append(mp3)
            if not cfg.wav:
                os.remove(wav_path)
            else:
                out_files.append(wav_path)
        else:
            out_files.append(wav_path)
        ch_path = f"{prefix}_narration_chapters{suffix}.txt"
        with open(ch_path, "w", encoding="utf-8") as f:
            f.write("\n".join(f"{stamp(o)}  [{n}] {pre}" for o, n, pre in chapters) + "\n")
        out_files.append(ch_path)
        log(f"Audio: {len(done)}/{len(chunks)} chunks, {stamp(dur(data))} -> {', '.join(out_files)}")
    report = [f"Narration report ({time.strftime('%Y-%m-%d %H:%M:%S')}, {(time.time() - started) / 60:.1f} min)",
              f"Model: {cfg.model} | voice: {cfg.voice} | style sent: {api.send_style} | gloss mode: {cfg.gloss}",
              f"Chunks: {len(done)}/{len(chunks)} complete | new API audio responses: {run.fresh} | splits: {run.splits}",
              "Requests per key: " + (", ".join(f"#{k + 1}: {c}" for k, c in sorted(api.used.items())) or "-"),
              "Keys rejected: " + (", ".join(f"#{k + 1}" for k in sorted(api.bad)) or "none")
              + " | daily quota used up: " + (", ".join(f"#{k + 1}" for k in sorted(api.exhausted)) or "none"),
              f"HTTP errors: {dict(api.errs) or 'none'} | time waiting for rate limits: {api.waited / 60:.1f} min",
              "Capability fallbacks: " + ("; ".join(api.quirks) or "none"),
              "Flags: " + ("; ".join(run.flags) or "none"),
              "Status: " + ("COMPLETE" if complete else f"INCOMPLETE - {stop}")]
    with open(f"{prefix}_narration_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    if not complete:
        log(f"INCOMPLETE: {len(chunks) - len(done)} chunk(s) left. Run again after the daily quota resets "
            f"(midnight Pacific time); finished chunks in '{cfg.workdir}' are reused.")
        return 2
    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
