"""
Professional, context-aware English -> Arabic translation of a transcript
(numbered .txt sentences, or .srt cues) using the Gemini API.

Usage:
    python translate_transcript.py <input_path> <output_prefix> [mode] [model]

    mode:  "arabic" (default) | "bilingual" | "both"
    model: primary Gemini model name (default: gemini-3.8-flash)

How it works
  1. CONTEXT PASS (one request over the whole transcript, any length):
     domain, summary, tone, a 100-150 term glossary and the speech-recognition
     mishearings that are evident from context (e.g. "coxone" -> COX-1).
  2. TRANSLATION. .txt input is grouped into PARAGRAPHS (~110 words) and
     translated paragraph by paragraph, so the Arabic reads as real prose, not
     as one line per audio fragment. .srt input stays 1:1 per cue (timing kept).
     Requests are id-matched; only missing/invalid ids are re-requested.
  3. TERMS: the model marks technical terms as  ⟦arabic|English⟧. The program
     (not the model) writes the English in parentheses on the FIRST appearance
     only, so it is never repeated and never forgotten.
  4. CHECKS (no API calls): number mismatch, possible omission/addition,
     leftover English, unmarked glossary terms, repeated phrases.
  4b. FREE-TIER AWARE: a lite-model answer is never accepted while a quality model
     can still answer. Unfinished batches are retried in rounds (pausing in between)
     until the translation stage's share of the time budget is used; only then do
     lite models answer, and those units are flagged, reviewed, and RE-TRANSLATED by a
     quality model on the next run (the checkpoint is kept while anything is unfinished).
  5. REVIEW: suspect units are re-read by a stronger model (Pro first, then the
     quality chain). Units answered by a lite model (or kept in English) are ALWAYS
     reviewed, with no cap; a failed review batch is retried once and the reason
     for any skipped/stopped review is written to the report.
  6. PROOFREAD: the whole Arabic text gets a language pass (~25 paragraphs per
     request) that returns tiny corrections (old -> new); the program validates
     (digits, markup and English terms must not change) and applies them.
  7. AUTO-TAG: glossary terms the model never marked get their English in
     parentheses at their first appearance (done by the program).
  8. TIMING: every stage's duration and request count are printed and reported.

Resilience
    * Quality chain (3.8 -> 3.6 -> 3.7). HTTP 503 means the MODEL is overloaded,
      not the key: no key rotation, the request moves to the next model at once.
      After 2 consecutive failures a model is demoted (tried last) for 10 minutes.
      Only if the whole chain fails do the "last resort" lite models answer, and
      every unit they touch is sent to review.
    * Sticky success: the model that answered last is tried first for STICKY_SECS
      (5 min) while it stays healthy; a failure of that model cancels the priority.
    * GEMMA 4 (middle tier, between Flash and lite): when the whole Flash chain fails twice in a
      row (2 rounds, ~2-3 min), the pending paragraphs go to gemma-4-31b-it -> gemma-4-26b-a4b-it
      in SMALL batches (~900 words, glossary filtered to the terms present in the batch) so the
      16K tokens/minute limit is respected (per-key sliding-window limiter). While Flash keeps
      failing, new batches try Gemma first for GEMMA_PREFER_SECS and Flash is probed again
      afterwards. Every Gemma paragraph is ALWAYS reviewed by Flash (no cap); if the review
      cannot run it stays flagged and is re-translated by Flash on the next run.
      Gemma capabilities are auto-calibrated: thinking low/medium -> minimal (high stays high),
      and on HTTP 400 the request degrades step by step (thinking, JSON schema, JSON mime,
      systemInstruction) - JSON is then enforced through the prompt and parsed leniently.
    * Batches are translated in parallel (TRANSLATE_PARALLEL workers, one key each
      at a time). The previous paragraphs given as context are then the English
      source (consistency comes from the glossary and the marked terms).
    * Several API keys are rotated (create each key in a DIFFERENT Google Cloud
      project: free-tier quotas are per project, not per key).
    * Per-day 429 -> that (key, model) pair is skipped for the run.
    * Truncated / blocked responses split the batch instead of blind retries.
    * Global deadline: partial output + checkpoint are saved on failure.

Environment variables
    GEMINI_API_KEY, GEMINI_API_KEY_2 ... GEMINI_API_KEY_8   (at least one)
    GEMINI_API_KEYS          comma-separated alternative to the above
    GEMINI_FALLBACK_MODELS   override the quality chain (comma-separated)
    GEMINI_LAST_RESORT_MODELS override the last-resort models
    GEMINI_THINKING_LEVEL    low (default, translation) | medium | high | off
                             (context pass and review always use high)
    GEMMA_MODELS             override the Gemma tier (comma-separated; default gemma-4-31b-it,
                             gemma-4-26b-a4b-it)
    TRANSLATE_GEMMA          on (default) | off   (Gemma middle tier)
    TRANSLATE_GEMMA_WORDS    English words per Gemma request (default 900)
    TRANSLATE_PARALLEL       max parallel requests (default: min(3, number of keys));
                             lowered automatically while models answer 503
    TRANSLATE_PROOFREAD      on (default) | off
    TRANSLATE_PROOF_BATCH    paragraphs per proofreading request (default 25)
    TRANSLATE_PROOF_LEVEL    thinking level of the proofreading pass (default medium)
    TRANSLATE_MAX_MINUTES    global time budget (default 60; stages get fixed shares of it)
    TRANSLATE_REVIEW_MODEL   default: the quality chain (pro-preview has no free-tier quota);
                             set a model name to try it first ; "off" disables review
    TRANSLATE_REVIEW_MAX     max NON-lite suspect units reviewed (default 15);
                             lite / failed units are always reviewed

Outputs (extension follows the input file):
    <prefix>_ar.<ext>          Arabic only
    <prefix>_bilingual.<ext>   English + Arabic
    <prefix>_report.txt        glossary, models used, review log, remaining flags
    <prefix>_progress.json     checkpoint (deleted on success; reused if present)

Uses only the Python standard library.
"""

import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------------------------------------------------ tunables
PARA_TARGET_WORDS = 110      # .txt: close a paragraph at a sentence end after this many words
PARA_MAX_WORDS = 220         # .txt: hard cap
CHUNK_WORDS = 2600           # .txt: English words per request
CHUNK_MAX_UNITS = 24
SRT_CHUNK_CUES = 40
CTX = {"txt": (2, 1), "srt": (6, 3)}   # (units before, units after) given as context

DEFAULT_MODEL = "gemini-3.8-flash"
QUALITY_CHAIN = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash"]
LAST_RESORT = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
DEFAULT_REVIEW_MODEL = ""          # empty = use the quality chain (free tier has no pro quota)

PER_MODEL_RETRIES = 4
CHAIN_ROUNDS = 2
ROUND_WAIT = 45
COOLDOWN_SECS = 60
MAX_SERVER_DELAY = 90
REQUEST_TIMEOUT = 420
DEFAULT_MAX_MINUTES = 60
QUALITY_PATIENCE = 240       # seconds spent on the quality chain per request
SHORT_PATIENCE = 45          # after 2 consecutive degradations
REVIEW_BATCH = 10
OVERLOAD_HTTP = {503}        # model overloaded: fail over at once, never rotate keys
DEMOTE_AFTER = 2             # consecutive failures before a model is demoted
DEMOTE_SECS = 600            # a demoted model is tried last for this long
STICKY_SECS = 300            # the model that just succeeded is tried FIRST for this long
PROOF_BATCH = 40
GEMMA = ["gemma-4-31b-it", "gemma-4-26b-a4b-it"]
GEMMA_WORDS = 900            # English words per Gemma request (16K tokens/min limit)
GEMMA_MAX_UNITS = {"txt": 8, "srt": 30}
GEMMA_TPM = 14_000           # input tokens per minute per key (limit 16K, safety margin)
GEMMA_RPM = 26               # requests per minute per key (limit 30)
GEMMA_PATIENCE = 150         # seconds per Gemma request (all Gemma models)
GEMMA_PREFER_SECS = 240      # after a Flash failure new batches try Gemma first for this long
CONTEXT_PATIENCE = 420
RETRY_PAUSE = (45, 75)       # seconds to pause between translation rounds when models are busy
# share of the time budget at which each stage must be finished
STAGE_FRAC = {"context": 0.20, "context_lite": 0.30, "translate": 0.55, "lite": 0.65, "review": 0.82, "proof": 0.95}
CONTEXT_CAP_CHARS = 600_000

RETRYABLE_HTTP = {429, 500, 502, 503, 504}
BLOCK_REASONS = {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
LATIN_RE = re.compile(r"[A-Za-z]")
MARK_RE = re.compile(r"⟦([^⟦⟧|]*)\|([^⟦⟧|]*)⟧")
STRAY_RE = re.compile(r"⟦([^⟦⟧]*)⟧")
REPEAT_RE = re.compile(r"(?<!\S)(\S+(?:\s+\S+){1,3})\s+\1(?!\S)")
SENT_END_RE = re.compile(r"[.?!\u2026][\"')\]]*$")
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# ------------------------------------------------------------- global state
KEYS = []
BAD_KEYS = set()
EXHAUSTED = set()        # (key_index, model) with no quota left for the run
KEY_COOL = {}            # (key_index, model) -> unix time
MODEL_COOL = {}
DEAD_MODELS = set()      # 404
NO_THINKING = set()
USED = Counter()
DEADLINE = [float("inf")]
RR = [0]
QUALITY_FAILS = [0]
LOCK = threading.RLock()
ABORT = [False]
PARALLEL = [1]
DEMOTED = {}             # model -> unix time until which it is tried last
MODEL_FAILS = Counter()  # consecutive failures per model
STICKY = [None, 0.0]     # [model that answered last, unix time until which it is tried first]
WAIT = [0.0]             # seconds spent sleeping (summed over threads)
TIMINGS = []             # (stage, seconds, requests)
STAGE_END = [float("inf")]
START = [0.0]
PAUSE_UNTIL = [0.0]      # shared pause: every worker waits (no thundering herd after 503s)
ERRS = Counter()         # (model, http code) -> count
NO_SCHEMA = set()        # models that rejected responseSchema
NO_MIME = set()          # models that rejected responseMimeType
NO_SYSTEM = set()        # models that rejected systemInstruction
QUIRK_LOG = []           # capability fallbacks discovered at run time (for the report)
GEMMA_PREFER = [0.0]     # unix time until which new batches try Gemma before Flash
GEMMA_PROBING = [False]  # True once Flash has failed: Flash is then probed with a single round
GEMMA_WIN = {}           # key index -> [(time, tokens)] sliding window for the Gemma limits
PROGRESS_PATH = [None]


class Gate:
    """Adaptive concurrency: lowered while models answer 503, restored after successes."""
    def __init__(self):
        self.cv = threading.Condition()
        self.limit = self.max = 1
        self.active = self.ok = 0
        self.last_cut = 0.0

    def setup(self, n):
        self.limit = self.max = max(1, n)

    def acquire(self, stop_at):
        with self.cv:
            while self.active >= self.limit:
                _guard(stop_at)
                self.cv.wait(1.0)
            self.active += 1

    def release(self):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()

    def overload(self):
        with self.cv:
            now = time.time()
            self.ok = 0
            if self.limit > 1 and now - self.last_cut >= 15:
                self.limit -= 1
                self.last_cut = now
                print(f"  concurrency lowered to {self.limit}")

    def success(self):
        with self.cv:
            self.ok += 1
            if self.ok >= 4 and self.limit < self.max:
                self.limit += 1
                self.ok = 0
                self.cv.notify_all()


GATE = Gate()


def set_stage_end(key):
    STAGE_END[0] = min(DEADLINE[0], START[0] + STAGE_FRAC[key] * (DEADLINE[0] - START[0]))


class Stage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        self.r0 = sum(USED.values())
        return self

    def __exit__(self, *exc):
        dt, req = time.time() - self.t0, sum(USED.values()) - self.r0
        TIMINGS.append((self.name, dt, req))
        print(f"  [time] {self.name}: {dt / 60:.1f} min, {req} request(s)")
        return False


def run_parallel(fn, items, workers, on_done):
    """Run fn(item) on up to `workers` threads; on_done(item, ok, value) runs in the main thread."""
    def wrap(it):
        try:
            return True, fn(it)
        except Exception as e:
            return False, e

    if workers <= 1 or len(items) <= 1:
        for it in items:
            ok, val = wrap(it)
            on_done(it, ok, val)
        return
    ex = ThreadPoolExecutor(max_workers=workers)
    futs = {ex.submit(wrap, it): it for it in items}
    try:
        for f in as_completed(futs):
            ok, val = f.result()
            on_done(futs[f], ok, val)
    except BaseException:
        ABORT[0] = True
        ex.shutdown(wait=False, cancel_futures=True)
        raise
    ex.shutdown(wait=True)


def is_gemma(model):
    return str(model).startswith("gemma")


def gemma_chain():
    return env_list("GEMMA_MODELS", GEMMA)


def gemma_allowed():
    if os.environ.get("TRANSLATE_GEMMA", "on").strip().lower() in ("off", "none", "0"):
        return False
    return chain_alive(gemma_chain())


def _gemma_throttle(ki, tokens, stop_at):
    """Per-key sliding window for Gemma (tokens and requests per minute); reserves capacity."""
    while True:
        with LOCK:
            now = time.time()
            win = GEMMA_WIN.setdefault(ki, [])
            win[:] = [(t, n) for t, n in win if now - t < 60]
            if not win or (sum(n for _, n in win) + tokens <= GEMMA_TPM and len(win) < GEMMA_RPM):
                win.append((now, tokens))
                return
            wait = win[0][0] + 60 - now
        _sleep(min(max(wait, 0.5) + 0.3, 8), stop_at)


def _degrade_quirk(model, raw, thinking_on, schema):
    """HTTP 400: switch off one unsupported feature. Non-Gemma models only lose thinking."""
    low = raw.lower()
    order = ["thinking", "schema", "mime", "system"]
    sets = {"thinking": NO_THINKING, "schema": NO_SCHEMA, "mime": NO_MIME, "system": NO_SYSTEM}
    active = {"thinking": thinking_on, "schema": bool(schema) and model not in NO_SCHEMA,
              "mime": model not in NO_MIME, "system": model not in NO_SYSTEM}
    hints = {"thinking": ("thinking", "thought"), "schema": ("schema",),
             "mime": ("mime", "json mode"), "system": ("system", "developer instruction")}
    cand = [k for k in order if active[k] and (is_gemma(model) or k == "thinking")]
    pick = next((k for k in cand if any(h in low for h in hints[k])), cand[0] if cand else None)
    if pick is None:
        return None
    sets[pick].add(model)
    QUIRK_LOG.append(f"{model}: تم رفض '{pick}' (HTTP 400) -> مُعطَّل لهذا النموذج")
    return pick


def note_failure(model):
    with LOCK:
        MODEL_FAILS[model] += 1
        if STICKY[0] == model:
            STICKY[0], STICKY[1] = None, 0.0
        if MODEL_FAILS[model] >= DEMOTE_AFTER:
            DEMOTED[model] = time.time() + DEMOTE_SECS
            if MODEL_FAILS[model] == DEMOTE_AFTER:
                print(f"  {model}: {DEMOTE_AFTER} failures in a row -> demoted for {DEMOTE_SECS // 60} min")


def note_success(model):
    with LOCK:
        USED[model] += 1
        MODEL_FAILS[model] = 0
        DEMOTED.pop(model, None)
        if not is_gemma(model):
            STICKY[0], STICKY[1] = model, time.time() + STICKY_SECS


class ApiUnavailable(Exception):
    def __init__(self, msg, fatal=False):
        super().__init__(msg)
        self.fatal = fatal


class StageTimeUp(ApiUnavailable):
    """The current stage used up its share of the time budget."""


class ResponseRejected(Exception):
    """The model answered but the output is unusable (truncated / blocked)."""


def env_list(name, default):
    raw = os.environ.get(name, "").strip()
    return [m.strip() for m in raw.split(",") if m.strip()] if raw else list(default)


def get_thinking_level():
    lvl = os.environ.get("GEMINI_THINKING_LEVEL", "low").strip().lower()
    return None if lvl in ("", "off", "none") else lvl


def load_keys():
    names = ["GEMINI_API_KEY"] + [f"GEMINI_API_KEY_{i}" for i in range(2, 9)]
    vals = [os.environ.get(n, "").strip() for n in names]
    vals += [v.strip() for v in os.environ.get("GEMINI_API_KEYS", "").split(",")]
    out = []
    for v in vals:
        if v and v not in out:
            out.append(v)
    return out


def build_chain(primary):
    chain = [primary]
    for m in env_list("GEMINI_FALLBACK_MODELS", QUALITY_CHAIN):
        if m not in chain:
            chain.append(m)
    return chain


def last_resort_chain():
    return env_list("GEMINI_LAST_RESORT_MODELS", LAST_RESORT)


def wc(text):
    return len(text.split())


# ---------------------------------------------------------------- parsing
def parse_transcript(path):
    """Return list of dicts: {idx, time (or None), text}."""
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read().replace("\r\n", "\n")
    segments = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [l for l in block.split("\n") if l.strip() != ""]
        if not lines:
            continue
        idx = lines[0].strip()
        rest = lines[1:]
        ts = None
        if rest and TIMESTAMP_RE.match(rest[0].strip()):
            ts = rest[0].strip()
            rest = rest[1:]
        text = " ".join(l.strip() for l in rest).strip()
        segments.append({"idx": idx, "time": ts, "text": text})
    return segments


def build_units(segments, kind):
    """srt -> one unit per cue.  txt -> paragraphs made of whole sentences."""
    units = []

    def add(group):
        text = " ".join(s["text"] for s in group if s["text"]).strip()
        units.append({
            "id": len(units) + 1,
            "en": text,
            "time": group[0]["time"] if kind == "srt" else None,
            "idx": group[0]["idx"],
            "words": wc(text),
        })

    if kind == "srt":
        for s in segments:
            add([s])
        return units
    cur, words = [], 0
    for s in segments:
        cur.append(s)
        words += wc(s["text"])
        if (words >= PARA_TARGET_WORDS and SENT_END_RE.search(s["text"])) or words >= PARA_MAX_WORDS:
            add(cur)
            cur, words = [], 0
    if cur:
        add(cur)
    return units


# ---------------------------------------------------------------- gemini
def _build_body(system, user, schema, thinking, model=""):
    gem = is_gemma(model)
    if gem and thinking:                       # Gemma 4: thinking is on/off only (high | minimal)
        thinking = "high" if thinking == "high" else "minimal"
    if schema and (gem or model in NO_SCHEMA):  # JSON also enforced through the prompt
        user = (user + "\n\nOUTPUT FORMAT: respond with ONLY valid JSON (no markdown fences, no commentary) "
                "matching this JSON schema:\n" + json.dumps(schema, ensure_ascii=False))
    gen = {}
    if model not in NO_MIME:
        gen["responseMimeType"] = "application/json"
    if schema and model not in NO_SCHEMA:
        gen["responseSchema"] = schema
    if thinking:
        gen["thinkingConfig"] = {"thinkingLevel": thinking}
    if model in NO_SYSTEM:
        body = {"contents": [{"role": "user", "parts": [{"text": system + "\n\n" + user}]}]}
    else:
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}]}
    if gen:
        body["generationConfig"] = gen
    return body


def _post(model, api_key, body):
    req = urllib.request.Request(
        API_URL.format(model=model),
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        out = json.loads(resp.read().decode("utf-8"))

    block = (out.get("promptFeedback") or {}).get("blockReason")
    if block:
        raise ResponseRejected(f"prompt blocked ({block})")
    cands = out.get("candidates") or []
    if not cands:
        raise ValueError("no candidates in response")
    cand = cands[0]
    finish = cand.get("finishReason")
    if finish == "MAX_TOKENS":
        raise ResponseRejected("output truncated (MAX_TOKENS)")
    if finish in BLOCK_REASONS:
        raise ResponseRejected(f"response blocked ({finish})")
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise ValueError(f"empty response (finishReason={finish})")
    return text


def _guard(stop_at):
    if ABORT[0]:
        raise ApiUnavailable("aborted", fatal=True)
    now = time.time()
    if now > DEADLINE[0]:
        raise ApiUnavailable("global time budget reached", fatal=True)
    if now > STAGE_END[0]:
        raise StageTimeUp("stage time budget reached", fatal=True)
    if now >= stop_at:
        raise ApiUnavailable("patience exhausted")


def _sleep(seconds, stop_at):
    _guard(stop_at)
    t0 = time.time()
    end = t0 + max(0.0, min(seconds, stop_at - t0))
    while not ABORT[0]:
        left = end - time.time()
        if left <= 0:
            break
        time.sleep(min(left, 1.0))
    with LOCK:
        WAIT[0] += time.time() - t0


def _backoff(attempt, stop_at):
    wait = min(5 * 2 ** (attempt - 1), 45)
    _sleep(wait + random.uniform(0, wait * 0.4), stop_at)


def _retry_delay(raw):
    m = (re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', raw)
         or re.search(r"retry in (\d+(?:\.\d+)?)s", raw))
    return float(m.group(1)) if m else None


def _wait_pause(stop_at):
    while True:
        left = PAUSE_UNTIL[0] - time.time()
        if left <= 0:
            return
        _sleep(min(left, 5), stop_at)


def chain_alive(chain):
    """False when every (key, model) pair of the chain has no quota left (or the model is gone)."""
    return any(m not in DEAD_MODELS and any(_key_ok(k, m) for k in range(len(KEYS))) for m in chain)


def _gated_post(model, key, body, stop_at):
    GATE.acquire(stop_at)
    try:
        return _post(model, key, body)
    finally:
        GATE.release()


def _key_ok(k, model):
    return k not in BAD_KEYS and (k, model) not in EXHAUSTED


def _any_key():
    return any(k not in BAD_KEYS for k in range(len(KEYS)))


def _pick_key(model, stop_at):
    """Round-robin over keys that still have quota for `model`; waits if all are cooling."""
    n = len(KEYS)
    while True:
        with LOCK:
            avail = [k for k in range(n) if _key_ok(k, model)]
            if not avail:
                return None
            now = time.time()
            ready = [k for k in avail if KEY_COOL.get((k, model), 0) <= now]
            if ready:
                for off in range(n):
                    k = (RR[0] + off) % n
                    if k in ready:
                        RR[0] = k + 1
                        return k
            wait = min(KEY_COOL.get((k, model), 0) for k in avail) - now
        _sleep(wait, stop_at)


def call_gemini(chain, system, user, schema=None, level="default", patience=None, rounds=None):
    """Walk `chain` (models) x keys until one answers. Returns (text, model)."""
    if level == "default":
        level = get_thinking_level()
    stop_at = min(DEADLINE[0], time.time() + patience) if patience else DEADLINE[0]
    last_err = None

    total_rounds = rounds or CHAIN_ROUNDS
    for rnd in range(1, total_rounds + 1):
        live = [m for m in chain if m not in DEAD_MODELS
                and any(_key_ok(k, m) for k in range(len(KEYS)))]
        if not live:
            raise ApiUnavailable(f"no usable model/key left ({last_err})", fatal=not _any_key())
        now = time.time()
        with LOCK:
            ready = [m for m in live if MODEL_COOL.get(m, 0) <= now and DEMOTED.get(m, 0) <= now]
            if STICKY[0] in ready and STICKY[1] > now:   # sticky success: the model that just worked goes first
                ready.remove(STICKY[0])
                ready.insert(0, STICKY[0])
        order = ready + [m for m in live if m not in ready]   # healthy models first

        for model in order:
            use_thinking = bool(level) and model not in NO_THINKING
            overloaded = False
            gem = is_gemma(model)
            max_att = PER_MODEL_RETRIES + (4 if gem else 0)   # room for the capability fallbacks
            for attempt in range(1, max_att + 1):
                _guard(stop_at)
                if not gem:                      # a Flash pause must not delay Gemma
                    _wait_pause(stop_at)
                ki = _pick_key(model, stop_at)
                if ki is None:
                    break
                body = _build_body(system, user, schema, level if use_thinking else None, model)
                if gem:
                    _gemma_throttle(ki, len(json.dumps(body, ensure_ascii=False)) // 3, stop_at)
                try:
                    text = _gated_post(model, KEYS[ki], body, stop_at)
                    note_success(model)
                    GATE.success()
                    return text, model
                except ResponseRejected:
                    raise
                except urllib.error.HTTPError as e:
                    raw = e.read().decode("utf-8", "ignore")
                    last_err = f"{model} (key #{ki + 1}) HTTP {e.code}: {raw[:160]}"
                    with LOCK:
                        ERRS[(model, e.code)] += 1
                    if e.code in (401, 403):
                        BAD_KEYS.add(ki)
                        print(f"  key #{ki + 1} rejected (HTTP {e.code}) -> not used again")
                        if not _any_key():
                            raise ApiUnavailable("every API key was rejected", fatal=True)
                        continue
                    if e.code == 404:
                        DEAD_MODELS.add(model)
                        print(f"  {model}: not found -> skipping model")
                        break
                    if e.code == 400:
                        q = _degrade_quirk(model, raw, use_thinking, schema)
                        if q:
                            if q == "thinking":
                                use_thinking = False
                            print(f"  {model}: {q} rejected (HTTP 400), retrying without it")
                            continue
                    if e.code in OVERLOAD_HTTP:
                        note_failure(model)
                        GATE.overload()
                        overloaded = True
                        print(f"  {model} overloaded (HTTP {e.code}) -> next model (keys are not rotated)")
                        break
                    if e.code == 429:
                        delay = _retry_delay(raw)
                        if "PerDay" in raw or "limit: 0" in raw or '"limit": 0' in raw \
                                or (delay is not None and delay > MAX_SERVER_DELAY):
                            with LOCK:
                                EXHAUSTED.add((ki, model))
                            print(f"  {model}: quota exhausted on key #{ki + 1} -> skipping that pair")
                            continue
                        with LOCK:
                            KEY_COOL[(ki, model)] = time.time() + (delay if delay is not None else 2 ** attempt) \
                                + random.uniform(1, 3)
                        continue  # _pick_key uses another key or waits
                    if e.code not in RETRYABLE_HTTP:
                        print(f"  {model}: non-retryable HTTP {e.code}, skipping model")
                        break
                except Exception as e:  # network, timeout, malformed/empty response
                    last_err = f"{model} {e!r}"

                if attempt < max_att:
                    print(f"  {model} retry {attempt}/{max_att - 1} ({last_err[:110]})")
                    _backoff(attempt, stop_at)

            MODEL_COOL[model] = time.time() + COOLDOWN_SECS
            if not overloaded:
                note_failure(model)
                print(f"  {model} unavailable -> trying next model")

        if rnd < total_rounds:
            wait = ROUND_WAIT + random.uniform(0, 15)
            print(f"  all models failed (round {rnd}/{CHAIN_ROUNDS}); waiting {wait:.0f}s")
            if not is_gemma(chain[0]):
                with LOCK:
                    PAUSE_UNTIL[0] = max(PAUSE_UNTIL[0], time.time() + wait)
                _wait_pause(stop_at)
            else:
                _sleep(wait, stop_at)
            MODEL_COOL.clear()

    raise ApiUnavailable(f"all models failed: {last_err}")


def call_quality(primary, system, user, schema=None, level="default", rounds=None):
    """Quality chain only (no lite fallback). Returns (text, model, False)."""
    text, model = call_gemini(build_chain(primary), system, user, schema, level, QUALITY_PATIENCE, rounds)
    QUALITY_FAILS[0] = 0
    return text, model, False


def call_tiered(primary, system, user, schema=None, level="default"):
    """Quality chain first; lite 'last resort' models only if it stays unavailable.
    Returns (text, model, degraded)."""
    patience = QUALITY_PATIENCE if QUALITY_FAILS[0] < 2 else SHORT_PATIENCE
    try:
        text, model = call_gemini(build_chain(primary), system, user, schema, level, patience)
        QUALITY_FAILS[0] = 0
        return text, model, False
    except ApiUnavailable as e:
        if e.fatal:
            raise
        QUALITY_FAILS[0] += 1
        print(f"  quality models unavailable ({str(e)[:100]}); using last-resort models "
              f"- these units will be flagged for review")
    text, model = call_gemini(last_resort_chain(), system, user, schema, level, None)
    return text, model, True


def clean_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:                      # preamble / trailing words around the JSON (Gemma)
        for o, c in (("[", "]"), ("{", "}")):
            a, b = text.find(o), text.rfind(c)
            if a != -1 and b > a:
                try:
                    return json.loads(text[a:b + 1])
                except ValueError:
                    pass
        raise


# ------------------------------------------------------------ markup helpers
def strip_markup(text):
    text = MARK_RE.sub(lambda m: m.group(1), text)
    text = STRAY_RE.sub(lambda m: m.group(1).split("|")[0], text)
    return text.replace("⟦", "").replace("⟧", "")


def fix_marks(raw):
    """Repair reversed markup: ⟦English|عربي⟧ -> ⟦عربي|English⟧."""
    def swap(m):
        a, b = m.group(1), m.group(2)
        if LATIN_RE.search(a) and not ARABIC_RE.search(a) and ARABIC_RE.search(b) and not LATIN_RE.search(b):
            return f"⟦{b.strip()}|{a.strip()}⟧"
        return m.group(0)
    return MARK_RE.sub(swap, raw)


def swap_known_english(job, raw):
    """English glossary terms left unmarked/untranslated in the Arabic -> ⟦glossary Arabic|English⟧."""
    if not LATIN_RE.search(raw):
        return raw
    kw = keep_words(job)
    with job.lock:
        pairs = list(job.gloss.values())
    table = {en.lower(): (en, ar) for en, ar in pairs
             if len(en) >= 4 and ARABIC_RE.search(ar) and not LATIN_RE.search(ar)
             and en.lower() not in kw and en.lower() != ar.lower()}
    if not table:
        return raw
    pat = re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(k) for k in sorted(table, key=len, reverse=True))
                     + r")(?![A-Za-z])", re.I)

    def repl(m):
        en, ar = table[m.group(1).lower()]
        before = m.string[m.start() - 1] if m.start() > 0 else ""
        if before == "ل" and ar.startswith("ال"):
            ar = ar[1:]
        return f"⟦{ar}|{en}⟧"

    parts = re.split(r"(⟦[^⟧]*⟧|\([^)]*\))", raw)
    for k in range(0, len(parts), 2):
        if LATIN_RE.search(parts[k]):
            parts[k] = pat.sub(repl, parts[k])
    return "".join(parts)


def render_text(text, seen):
    """⟦ar|en⟧ -> 'ar (en)' on the first appearance of `en`, else 'ar'."""
    def repl(m):
        ar, en = m.group(1).strip(), m.group(2).strip()
        if not ar:
            return en
        key = en.lower()
        if not LATIN_RE.search(en) or key in seen:
            return ar
        seen.add(key)
        if re.match(r"\s*\(\s*[A-Za-z]", m.string[m.end():m.end() + 40]):
            return ar  # the model already wrote the English itself
        return f"{ar} ({en})"

    out = strip_markup(MARK_RE.sub(repl, text))
    out = re.sub(r"[ \t]+", " ", out).strip()
    for _ in range(2):
        out = REPEAT_RE.sub(r"\1", out)
    return out


# ---------------------------------------------------------------- job state
class Job:
    def __init__(self, units, kind, primary):
        self.units = units
        self.kind = kind
        self.primary = primary
        n = len(units)
        self.ar = [None] * n          # raw Arabic (with markup)
        self.deg = [False] * n        # answered by a last-resort model
        self.gem = [False] * n        # answered by Gemma (mid tier): must be reviewed by Flash
        self.carry = set()            # Gemma units from a checkpoint: re-translated by Flash this run
        self.model = [""] * n
        self.kept_en = [False] * n
        self.reviewed = {}            # id -> "changed" | "unchanged"
        self.brief = {"domain": "", "summary": "", "tone": "", "glossary": [], "asr_corrections": [],
                      "keep_english": []}
        self.lock = threading.RLock()
        self.lite_ok = False          # lite models may answer (last-resort pass only)
        self.proofed = set()          # unit ids already proofread
        self.autotag_log = []
        self.proof_log = []
        self.proof_stats = {"applied": 0, "rejected": 0, "failed_batches": 0}
        self.gloss = {}               # english.lower() -> (english, arabic)
        self.core = set()             # keys that came from the context pass
        self.review_log = []

    def fingerprint(self):
        h = hashlib.sha256()
        h.update(self.kind.encode())
        for u in self.units:
            h.update(u["en"].encode("utf-8"))
        return h.hexdigest()[:16]

    def add_brief_glossary(self):
        for g in self.brief.get("glossary", []):
            en, ar = (g.get("en") or "").strip(), (g.get("ar") or "").strip()
            if en and ar and en.lower() not in self.gloss:
                self.gloss[en.lower()] = (en, ar)
                self.core.add(en.lower())

    def learn_terms(self, raw_ar):
        with self.lock:
            for ar, en in MARK_RE.findall(raw_ar):
                en, ar = en.strip(), ar.strip()
                if en and ar and LATIN_RE.search(en) and en.lower() not in self.gloss:
                    self.gloss[en.lower()] = (en, ar)


# ---------------------------------------------------------------- context pass
CONTEXT_SYSTEM = (
    "You are a senior translator and subject-matter editor preparing an "
    "English-to-Arabic translation of an automatically transcribed spoken "
    "transcript (podcast/lecture). The transcript comes from speech recognition "
    "and may contain mishearings of specialised words and names."
)

CONTEXT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "domain": {"type": "STRING"},
        "summary": {"type": "STRING"},
        "tone": {"type": "STRING"},
        "glossary": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"en": {"type": "STRING"}, "ar": {"type": "STRING"}},
                "required": ["en", "ar"],
            },
        },
        "asr_corrections": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"heard": {"type": "STRING"}, "meant": {"type": "STRING"}},
                "required": ["heard", "meant"],
            },
        },
        "keep_english": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["domain", "summary", "tone", "glossary", "asr_corrections"],
}


def sample_for_context(units):
    full = "\n".join(u["en"] for u in units)
    if len(full) <= CONTEXT_CAP_CHARS:
        return full
    third = CONTEXT_CAP_CHARS // 3
    mid = len(full) // 2
    return "\n[...]\n".join([full[:third], full[mid - third // 2: mid + third // 2], full[-third:]])


def build_brief(job):
    prompt = (
        "Read this full transcript and return JSON with:\n"
        "- domain: the subject field and sub-field (e.g. medicine / hematology).\n"
        "- summary: 3-5 sentences on what it covers, its structure and the speakers' dynamic.\n"
        "- tone: the Arabic register to use (e.g. educational, conversational, for medical students).\n"
        "- glossary: 100-150 key terms with the single best standard Arabic rendering each: diseases, "
        "signs and symptoms, anatomy, physiology, lab tests, drugs, eponyms, acronyms, person names and "
        "recurring expressions/analogies. Use the established Arabic term from the Unified Medical "
        "Dictionary / Arabic medical textbooks when one exists, otherwise a short descriptive Arabic phrase "
        "- never an invented transliteration. 'en' must be the CORRECT English spelling.\n"
        "- asr_corrections: words that speech recognition clearly misheard, ONLY when the intended word is "
        "evident from the surrounding context (e.g. heard 'coxone' -> meant 'COX-1'). Leave empty if none.\n"
        "- keep_english: words or names that must stay in English inside the Arabic text, e.g. the words of "
        "a memory trick / mnemonic (such as a made-up name or phrase used to remember a list) and brand names. "
        "Leave empty if none.\n\n"
        f"TRANSCRIPT:\n{sample_for_context(job.units)}"
    )
    try:
        try:
            text, _ = call_gemini(build_chain(job.primary), CONTEXT_SYSTEM, prompt, CONTEXT_SCHEMA,
                                  level="high", patience=CONTEXT_PATIENCE)
        except ApiUnavailable as e:
            if e.fatal and not isinstance(e, StageTimeUp):
                raise
            print("  quality models unavailable for the context pass -> using lite models (weaker glossary)")
            set_stage_end("context_lite")
            text, _, _ = call_tiered(job.primary, CONTEXT_SYSTEM, prompt, CONTEXT_SCHEMA, level="high")
        brief = clean_json(text)
        for k, v in (("domain", ""), ("summary", ""), ("tone", ""), ("glossary", []), ("asr_corrections", []),
                     ("keep_english", [])):
            brief.setdefault(k, v)
        return brief
    except (SystemExit, ApiUnavailable):
        raise
    except Exception as e:  # bad JSON / rejected output: continue without a brief
        print(f"  (context pass failed, continuing without brief: {e!r})")
        return {"domain": "", "summary": "", "tone": "", "glossary": [], "asr_corrections": [],
                "keep_english": []}


# ---------------------------------------------------------------- translation
SYSTEM_COMMON = """You are a professional English-to-Arabic translator and medical editor specialising in educational spoken content.
Rules:
1. Write clear, natural Modern Standard Arabic (فصحى مبسطة) for students, as if originally written in Arabic: complete, well-formed sentences with correct grammar and agreement. Keep the speakers' meaning, tone and emphasis. Drop empty fillers (you know, like, right?, yeah) unless they carry meaning.
2. Translate by MEANING and CONTEXT. Use the glossary, summary and neighbouring items to resolve pronouns, ambiguity and idioms. Never translate an idiom or analogy literally; render its intended meaning.
3. The English comes from AUTOMATIC SPEECH RECOGNITION and may contain mishearings of medical terms and names. Use the domain, the known mishearings and the context to infer the intended word and translate THAT. Never translate a nonsense word literally and never invent a term. If two readings are possible, choose the one that fits the medical context.
4. Terminology: use the standard Arabic term used in Arabic medical education (Unified Medical Dictionary / textbooks). Follow the glossary exactly and keep ONE Arabic term per concept for the whole document. If no standard term exists, use a short descriptive Arabic phrase instead of a transliteration.
5. TERM MARKUP: every time you use a glossary term, or any other technical term (disease, sign, anatomical structure, cell, drug, test, eponym, person name, acronym), write it as ⟦arabic|English⟧. `arabic` is the exact inflected form needed in the sentence (put attached prefixes such as و ب ل ف outside the brackets); `English` is the canonical English term, using the glossary's English or the corrected spelling when it was misheard. For an acronym write the acronym as the arabic part, e.g. ⟦VWF|von Willebrand factor⟧. Do NOT write English in parentheses yourself - the program adds it on first appearance only. Do not mark ordinary words.
6. Numbers, units, doses, ages and lab values must be exact; write them with digits (0-9). Keep nothing in English except inside the markup and the words listed under KEEP IN ENGLISH (leave those exactly as given, in Latin letters).
7. Never omit, merge, split, summarise or add content. Output EXACTLY one object {"i": same id, "ar": Arabic} per input item, every id once. Return only a JSON array."""

SYSTEM_TXT = SYSTEM_COMMON + """
8. Each item is a whole PARAGRAPH of the lecture: translate it into ONE flowing Arabic paragraph (several sentences are fine). Join short interjections of the second speaker naturally into the flow."""

SYSTEM_SRT = SYSTEM_COMMON + """
8. Each item is ONE subtitle cue, often a fragment that starts or ends mid-sentence. Keep a strict 1:1 alignment: translate the fragment as the matching fragment of the Arabic sentence, keep each cue short, and move no content between cues beyond what Arabic word order forces. A short interjection ("Yeah.", "Right.") gets a short natural Arabic equivalent."""

CHUNK_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {"i": {"type": "INTEGER"}, "ar": {"type": "STRING"}},
        "required": ["i", "ar"],
        "propertyOrdering": ["i", "ar"],
    },
}


def brief_text(job):
    b = job.brief
    return (f"Domain: {b.get('domain', '')}\nSummary: {b.get('summary', '')}\n"
            f"Arabic tone: {b.get('tone', '')}")


def glossary_text(job, only_for=None):
    with job.lock:
        pairs = list(job.gloss.values())
    if only_for is not None:                 # Gemma: only the terms present in this batch
        low = only_for.lower()
        pairs = [(en, ar) for en, ar in pairs if en.lower() in low]
    return "\n".join(f"- {en} -> {ar}" for en, ar in pairs) or "(none)"


def keep_text(job):
    return ", ".join(str(x) for x in job.brief.get("keep_english", []) if x) or "(none)"


def corrections_text(job, only_for=None):
    low = only_for.lower() if only_for is not None else None
    rows = [f"- {c.get('heard', '')} -> {c.get('meant', '')}" for c in job.brief.get("asr_corrections", [])
            if c.get("heard") and c.get("meant") and (low is None or str(c["heard"]).lower() in low)]
    return "\n".join(rows) or "(none)"


def parse_items(result, pending):
    """Accept valid items one by one. Returns ({id: ar}, [problems])."""
    if isinstance(result, dict):
        result = result.get("units") or result.get("items") or []
    if not isinstance(result, list):
        return {}, ["response is not a list"]
    by_id = {u["id"]: u for u in pending}
    got, problems = {}, []
    for r in result:
        if not isinstance(r, dict) or not isinstance(r.get("i"), int) or not isinstance(r.get("ar"), str):
            problems.append("malformed item")
            continue
        i = r["i"]
        if i not in by_id:
            problems.append(f"unknown id {i}")
            continue
        if i in got:
            continue
        ar, en = r["ar"].strip(), by_id[i]["en"]
        if en.strip() and not ar:
            problems.append(f"empty translation for {i}")
            continue
        if len(en) >= 25 and not ARABIC_RE.search(strip_markup(ar)):
            problems.append(f"id {i} left untranslated")
            continue
        got[i] = ar
    missing = [u["id"] for u in pending if u["id"] not in got]
    if missing:
        problems.append(f"missing ids {missing[:8]}{'...' if len(missing) > 8 else ''}")
    return got, problems


def context_lines(job, idx_range, with_ar):
    rows = []
    for i in idx_range:
        u = job.units[i]
        if with_ar:
            if job.ar[i] is None:
                continue
            rows.append(f"EN: {u['en']}\nAR: {strip_markup(job.ar[i])}")
        else:
            rows.append(u["en"])
    return "\n".join(rows)


def needs_work(job, k, redo):
    return job.ar[k] is None or (redo and (job.deg[k] or job.kept_en[k] or k in job.carry))


def gemma_fallback(job, pending):
    """Translate `pending` with Gemma in small batches. Returns the units still needing work."""
    todo = [u for u in pending if not (job.gem[u["id"] - 1] and job.ar[u["id"] - 1] is not None)]
    if not todo or not gemma_allowed():
        return pending
    cap = GEMMA_MAX_UNITS[job.kind]
    try:
        words = max(300, int(os.environ.get("TRANSLATE_GEMMA_WORDS", GEMMA_WORDS)))
    except ValueError:
        words = GEMMA_WORDS
    subs, cur, w = [], [], 0
    for u in todo:
        if cur and (len(cur) >= cap or w + u["words"] > words):
            subs.append(cur)
            cur, w = [], 0
        cur.append(u)
        w += u["words"]
    if cur:
        subs.append(cur)
    print(f"  Gemma tier: {len(todo)} unit(s) in {len(subs)} small request(s) "
          f"({' -> '.join(gemma_chain())}); every Gemma unit is reviewed by Flash afterwards")
    for sub in subs:
        try:
            translate_batch(job, sub, gemma=True)
        except ApiUnavailable as e:
            if e.fatal:
                raise
            print(f"  Gemma unavailable too: {str(e)[:100]}")
            break
        if PROGRESS_PATH[0]:
            save_progress(PROGRESS_PATH[0], job)
    return [u for u in pending if needs_work(job, u["id"] - 1, True)]


def translate_batch(job, batch, gemma=False):
    """Translate `batch` (list of units). Re-requests only what is missing; splits on repeated failure.
    gemma=True: Gemma request (small batch, filtered glossary, units flagged for Flash review)."""
    system = SYSTEM_SRT if job.kind == "srt" else SYSTEM_TXT
    nb, na = CTX[job.kind]
    pending = list(batch)
    if not gemma and gemma_allowed() and (job.lite_ok or time.time() < GEMMA_PREFER[0]):
        pending = gemma_fallback(job, pending)      # Flash keeps failing: Gemma goes first for a while
        if not pending:
            return

    for attempt in range(1, 4):
        a, b = pending[0]["id"] - 1, pending[-1]["id"] - 1
        arabic_ctx = PARALLEL[0] <= 1
        ctx_prev = context_lines(job, range(max(0, a - nb), a), arabic_ctx)
        prev_label = ("PREVIOUS ITEMS (already translated, for continuity only)" if arabic_ctx else
                      "PREVIOUS ITEMS (English source, context only - do NOT translate; "
                      "stay consistent through the GLOSSARY)")
        ctx_next = context_lines(job, range(b + 1, min(len(job.units), b + 1 + na)), False)
        items = json.dumps([{"i": u["id"], "en": u["en"]} for u in pending], ensure_ascii=False)
        src = " ".join(u["en"] for u in pending) if gemma else None
        prompt = (
            f"{brief_text(job)}\n\n"
            f"GLOSSARY (use these Arabic renderings exactly):\n{glossary_text(job, src)}\n\n"
            f"KNOWN TRANSCRIPT MISHEARINGS (heard -> meant):\n{corrections_text(job, src)}\n\n"
            f"KEEP IN ENGLISH (leave exactly as written):\n{keep_text(job)}\n\n"
            f"{prev_label}:\n{ctx_prev or '(none)'}\n\n"
            f"UPCOMING ITEMS (context only, do NOT translate):\n{ctx_next or '(none)'}\n\n"
            f"TRANSLATE these {len(pending)} items. Return a JSON array of exactly {len(pending)} objects "
            f'{{"i", "ar"}} using the same i values:\nITEMS:\n{items}'
        )
        try:
            if gemma:
                text, model = call_gemini(gemma_chain(), system, prompt, CHUNK_SCHEMA,
                                          patience=GEMMA_PATIENCE, rounds=1)
                degraded = False
            elif job.lite_ok:
                text, model, degraded = call_tiered(job.primary, system, prompt, CHUNK_SCHEMA)
            else:
                text, model, degraded = call_quality(job.primary, system, prompt, CHUNK_SCHEMA,
                                                     rounds=1 if GEMMA_PROBING[0] else None)
            result = clean_json(text)
        except ApiUnavailable as e:
            if e.fatal or gemma or job.lite_ok or not gemma_allowed():
                raise
            GEMMA_PROBING[0] = True
            GEMMA_PREFER[0] = time.time() + GEMMA_PREFER_SECS
            print("  Flash chain failed -> Gemma tier takes over for these units")
            pending = gemma_fallback(job, pending)
            if not pending:
                return
            raise
        except SystemExit:
            raise
        except ResponseRejected as e:
            print(f"  response rejected: {e}")
            break  # retrying the same batch will not help; split it
        except Exception as e:
            print(f"  parse error attempt {attempt}/3: {e!r}")
            continue

        got, problems = parse_items(result, pending)
        for i, ar in got.items():
            ar = swap_known_english(job, fix_marks(ar))
            job.ar[i - 1] = ar
            job.kept_en[i - 1] = False
            job.reviewed.pop(i, None)
            job.proofed.discard(i)
            job.deg[i - 1] = degraded
            job.gem[i - 1] = gemma
            job.carry.discard(i - 1)
            job.model[i - 1] = model
            job.learn_terms(ar)
        pending = [u for u in pending if u["id"] not in got]
        if not pending:
            return
        print(f"  invalid/missing: {'; '.join(problems[:3])} (attempt {attempt}/3)")

    if gemma:                                   # Gemma never keeps English: Flash gets the leftovers
        if len(pending) > 1:
            mid = len(pending) // 2
            translate_batch(job, pending[:mid], gemma=True)
            translate_batch(job, pending[mid:], gemma=True)
        return
    if len(pending) == 1:
        u = pending[0]
        print(f"  WARNING: unit {u['id']} could not be translated; keeping English.")
        job.ar[u["id"] - 1] = u["en"]
        job.kept_en[u["id"] - 1] = True
        return
    mid = len(pending) // 2
    print(f"  splitting {len(pending)} units into halves")
    translate_batch(job, pending[:mid])
    translate_batch(job, pending[mid:])


def plan_chunks(job, redo_lite=False):
    """Units still to translate (and, with redo_lite, those that only have a lite/English draft)."""
    chunks, cur, words = [], [], 0
    limit_units = SRT_CHUNK_CUES if job.kind == "srt" else CHUNK_MAX_UNITS
    for u in job.units:
        k = u["id"] - 1
        if not needs_work(job, k, redo_lite):
            continue
        cur.append(u)
        words += u["words"]
        if len(cur) >= limit_units or (job.kind == "txt" and words >= CHUNK_WORDS):
            chunks.append(cur)
            cur, words = [], 0
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------- checks
def to_western(text):
    return text.translate(DIGITS)


NUM_STEMS = {0: "صفر", 1: ("واحد", "احد", "إحد", "أحد"), 2: ("اثن", "إثن", "ثنت", "ثنين"), 3: "ثلاث", 4: "أربع",
             5: "خمس", 6: "ست", 7: "سبع", 8: "ثمان", 9: "تسع", 10: "عشر"}
TENS_STEMS = {2: "عشر", 3: "ثلاث", 4: "أربع", 5: "خمس", 6: "ست", 7: "سبع", 8: "ثمان", 9: "تسع"}
AR_PREFIX_RE = re.compile(r"^[وفبلك]?(?:ال)?")


def _stems(v):
    return (v,) if isinstance(v, str) else v


def number_in_words(num, plain):
    """True if the integer `num` (0-99) appears to be written in Arabic words in `plain`."""
    if not num.isdigit() or int(num) > 99:
        return False
    toks = [AR_PREFIX_RE.sub("", t) for t in re.findall(r"[\u0600-\u06FF]+", plain)]
    toks += [t for t in re.findall(r"[\u0600-\u06FF]+", plain)]

    def has(stems):
        return any(t.startswith(st) for t in toks for st in _stems(stems))

    n = int(num)
    if n <= 10:
        return has(NUM_STEMS[n])
    if n < 20:
        return has("عشر") and has(NUM_STEMS[n - 10])
    tens, unit = n // 10, n % 10
    stem = TENS_STEMS[tens]
    if not has(stem + "و") and not has(stem + "ين") and not has(stem + "ون"):
        if not (tens == 2 and has("عشر")):
            return False
    return unit == 0 or has(NUM_STEMS[unit])


def keep_words(job):
    out = set()
    for x in job.brief.get("keep_english", []):
        out |= {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]*", str(x))}
    return out


def check_unit(job, u):
    """Deterministic quality flags for one unit (no API calls)."""
    i = u["id"] - 1
    if job.ar[i] is None:
        return ["not translated"]
    raw = job.ar[i] or ""
    plain = strip_markup(raw)
    en = u["en"]
    reasons = []
    if job.kept_en[i]:
        return ["could not be translated"]
    if job.deg[i]:
        reasons.append("answered by a fallback (lite) model")
    if job.gem[i]:
        reasons.append("translated by Gemma (mid tier): check terminology, meaning, omissions and grammar")
    ew, aw = wc(en), wc(plain)
    if ew >= 12:
        ratio = aw / ew
        if ratio < 0.6:
            reasons.append(f"possible omission (Arabic/English length {ratio:.2f})")
        elif ratio > 1.7:
            reasons.append(f"possible addition (Arabic/English length {ratio:.2f})")
    outside = re.sub(r"\([^)]*\)", "", plain)
    outside = re.sub(r"«[^»]*»|“[^”]*”|\"[^\"]*\"",
                     lambda m: "" if LATIN_RE.search(m.group(0)) and not ARABIC_RE.search(m.group(0)) else m.group(0),
                     outside)
    kw = keep_words(job)
    left = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", outside)
            if not w.isupper() and w.lower() not in kw]
    if len(left) >= 2:
        reasons.append("untranslated English: " + ", ".join(left[:4]))
    en_nums = set(re.findall(r"\d+(?:[.,]\d+)?", en))
    ar_nums = set(re.findall(r"\d+(?:[.,]\d+)?", to_western(plain)))
    miss = {n for n in en_nums - ar_nums
            if len(n) > 1 and not number_in_words(n, plain)}   # single digits / numbers in words are not flagged
    if miss:
        reasons.append("numbers not found in Arabic: " + ", ".join(sorted(miss)[:4]))
    marked = {en_k.strip().lower() for _, en_k in MARK_RE.findall(raw)}
    unmarked = [k for k in job.core
                if k not in marked and re.search(r"(?<![A-Za-z])" + re.escape(k) + r"(?![A-Za-z])", en, re.I)]
    if len(unmarked) >= 2:
        reasons.append("glossary terms not marked: " + ", ".join(unmarked[:3]))
    return reasons


# ---------------------------------------------------------------- review
REVIEW_SYSTEM = """You are a senior English-to-Arabic medical translator reviewing drafts of a transcribed lecture.
For every item you get: the English source (automatic transcript - it may contain mishearings), the current Arabic draft (with ⟦arabic|English⟧ term markup) and the issues that were detected.
Fix: mistranslations, omissions, additions, speech-recognition mishearings, non-standard or inconsistent terminology (follow the glossary exactly), ungrammatical or awkward Arabic (verb forms, gender/number agreement, odd phrases), leftover English, wrong numbers. Words under KEEP IN ENGLISH must stay in English.
Keep the markup rules: every technical term stays wrapped as ⟦arabic|English⟧ and no English is written in parentheses by you.
If a draft is already correct, return it unchanged. Never summarise, never add content.
Return ONLY a JSON array with one {"i": same id, "ar": final Arabic} per item."""


def _review_worker(job, chain, part):
    items = json.dumps([
        {"i": u["id"], "en": u["en"], "draft": job.ar[u["id"] - 1], "issues": r} for u, r in part
    ], ensure_ascii=False)
    prompt = (
        f"{brief_text(job)}\n\nGLOSSARY:\n{glossary_text(job)}\n\n"
        f"KNOWN TRANSCRIPT MISHEARINGS (heard -> meant):\n{corrections_text(job)}\n\n"
        f"KEEP IN ENGLISH (do not translate):\n{keep_text(job)}\n\n"
        f"REVIEW these {len(part)} items:\nITEMS:\n{items}"
    )
    last = None
    for _ in range(2):                      # a failed batch is retried once
        try:
            text, model = call_gemini(chain, REVIEW_SYSTEM, prompt, CHUNK_SCHEMA,
                                      level="high", patience=QUALITY_PATIENCE)
            got, problems = parse_items(clean_json(text), [u for u, _r in part])
            return got, problems, model
        except ApiUnavailable as e:
            if e.fatal:
                raise
            last = e
        except Exception as e:
            last = e
    raise RuntimeError(f"failed twice: {str(last)[:140]}")


def review_pass(job):
    rm = os.environ.get("TRANSLATE_REVIEW_MODEL", DEFAULT_REVIEW_MODEL).strip()
    if rm.lower() in ("off", "none", "0"):
        print("Review pass disabled.")
        job.review_log.append("المراجعة معطّلة (TRANSLATE_REVIEW_MODEL=off)")
        return
    try:
        cap = int(os.environ.get("TRANSLATE_REVIEW_MAX", "15"))
    except ValueError:
        cap = 15
    suspects = []
    for u in job.units:
        if job.ar[u["id"] - 1] is None or u["id"] in job.reviewed:
            continue
        reasons = check_unit(job, u)
        if reasons:
            suspects.append((u, reasons))
    if not suspects:
        print("Checks: no suspect units.")
        return
    must = [t for t in suspects if job.deg[t[0]["id"] - 1] or job.gem[t[0]["id"] - 1]
            or job.kept_en[t[0]["id"] - 1]]
    others = sorted((t for t in suspects if t not in must), key=lambda t: -len(t[1]))
    chosen = must + others[:cap]
    print(f"Checks: {len(suspects)} suspect unit(s); reviewing {len(chosen)} "
          f"({len(must)} lite/Gemma/failed always, up to {cap} others).")
    if len(others) > cap:
        job.review_log.append(f"{len(others) - cap} وحدة مشتبه بها أقل أولوية لم تُراجَع (الحد الأقصى {cap})")
    chain = ([rm] if rm else []) + [m for m in build_chain(job.primary) if m != rm]
    parts = [chosen[k:k + REVIEW_BATCH] for k in range(0, len(chosen), REVIEW_BATCH)]
    stopped = []

    def on_done(part, ok, val):
        ids = [u["id"] for u, _ in part]
        if not ok:
            reason = str(val)[:140]
            job.review_log.append(f"الوحدات {ids}: لم تتم المراجعة ({reason})")
            print(f"  review batch {ids} skipped: {reason}")
            if isinstance(val, ApiUnavailable) and val.fatal:
                stopped.append(reason)
            return
        got, problems, model = val
        for i, ar in got.items():
            ar = swap_known_english(job, fix_marks(ar))
            old = job.ar[i - 1]
            changed = strip_markup(ar) != strip_markup(old)
            job.ar[i - 1] = ar
            job.deg[i - 1] = False                       # a non-lite model has now read it
            job.gem[i - 1] = False                       # reviewed by Flash
            if ARABIC_RE.search(strip_markup(ar)):
                job.kept_en[i - 1] = False
            job.reviewed[i] = "changed" if changed else "unchanged"
            job.learn_terms(ar)
            job.review_log.append(f"الوحدة {i}: {'عُدّلت' if changed else 'أُكِّدت دون تغيير'} بواسطة {model}")
        missing = [i for i in ids if i not in got]
        if missing:
            job.review_log.append(f"الوحدات {missing}: لم يُرجعها المراجِع ({'; '.join(problems[:2])})")
            print(f"  review: {'; '.join(problems[:2])}")

    run_parallel(lambda part: _review_worker(job, chain, part), parts, min(PARALLEL[0], len(parts)), on_done)
    if stopped:
        job.review_log.append(f"توقفت المراجعة مبكرًا: {stopped[0]}")
    left = [u["id"] for u in job.units if job.deg[u["id"] - 1]]
    if left:
        job.review_log.append(f"وحدات ما زالت مترجمة بنموذج lite بعد المراجعة: {left}")
    gleft = [u["id"] for u in job.units if job.gem[u["id"] - 1]]
    if gleft:
        job.review_log.append(f"وحدات Gemma التي لم تُراجَع (ستُعاد ترجمتها بنموذج Flash في التشغيل القادم): {gleft}")


# ---------------------------------------------------------------- proofreading
PROOF_SYSTEM = """You are a meticulous Arabic copy-editor (Modern Standard Arabic) proofreading the Arabic translation of a medical lecture.
Each item is one paragraph with ⟦arabic|English⟧ term markup. Find ONLY real language errors: wrong verb forms, gender/number/definiteness agreement, wrong prepositions, ungrammatical or meaningless phrases, unnatural calques from English (e.g. odd collocations), typos.
Do NOT change meaning, terminology, numbers, English text or the ⟦ | ⟧ markup, and do not restyle sentences that are already correct.
Return a JSON array of corrections {"i": paragraph id, "old": exact short substring copied from that paragraph (a word or phrase, at most 8 words, appearing exactly once in it), "new": the corrected replacement}. If a paragraph is fine, return nothing for it. If nothing needs fixing, return []."""

PROOF_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {"i": {"type": "INTEGER"}, "old": {"type": "STRING"}, "new": {"type": "STRING"}},
        "required": ["i", "old", "new"],
        "propertyOrdering": ["i", "old", "new"],
    },
}


def _correction_ok(old, new):
    if not old or old == new or len(new) > len(old) * 3 + 20:
        return False
    if re.findall(r"\d+", to_western(old)) != re.findall(r"\d+", to_western(new)):
        return False
    if re.findall(r"[A-Za-z][A-Za-z'-]*", old) != re.findall(r"[A-Za-z][A-Za-z'-]*", new):
        return False
    return all(old.count(c) == new.count(c) for c in "⟦⟧|")


def _proof_worker(job, chain, level, batch):
    items = json.dumps([{"i": u["id"], "ar": job.ar[u["id"] - 1]} for u in batch], ensure_ascii=False)
    prompt = (f"Domain: {job.brief.get('domain', '')}\n"
              f"KEEP IN ENGLISH: {keep_text(job)}\n\n"
              f"PROOFREAD these {len(batch)} paragraphs:\nITEMS:\n{items}")
    last = None
    for _ in range(2):
        try:
            text, model = call_gemini(chain, PROOF_SYSTEM, prompt, PROOF_SCHEMA, level=level,
                                      patience=QUALITY_PATIENCE)
            res = clean_json(text)
            if isinstance(res, dict):
                res = res.get("corrections") or res.get("items") or []
            if not isinstance(res, list):
                raise ValueError("response is not a list")
            return res, model
        except ApiUnavailable as e:
            if e.fatal:
                raise
            last = e
        except Exception as e:
            last = e
    raise RuntimeError(f"failed twice: {str(last)[:140]}")


def proofread_pass(job):
    if os.environ.get("TRANSLATE_PROOFREAD", "on").strip().lower() in ("off", "none", "0"):
        print("Proofreading pass disabled.")
        return
    try:
        size = max(5, int(os.environ.get("TRANSLATE_PROOF_BATCH", PROOF_BATCH)))
    except ValueError:
        size = PROOF_BATCH
    lvl = os.environ.get("TRANSLATE_PROOF_LEVEL", "medium").strip().lower()
    level = None if lvl in ("", "off", "none") else lvl
    todo = [u for u in job.units if job.ar[u["id"] - 1] and not job.kept_en[u["id"] - 1]
            and u["id"] not in job.proofed]
    batches = [todo[k:k + size] for k in range(0, len(todo), size)]
    chain = build_chain(job.primary)
    print(f"Proofreading {len(todo)} paragraph(s) in {len(batches)} request(s)...")
    stats = job.proof_stats

    def on_done(batch, ok, val):
        ids = [u["id"] for u in batch]
        if not ok:
            stats["failed_batches"] += 1
            job.proof_log.append(f"الدفعة {ids[0]}-{ids[-1]}: لم يتم التدقيق اللغوي ({str(val)[:140]})")
            print(f"  proofreading batch {ids[0]}-{ids[-1]} skipped: {str(val)[:100]}")
            return
        corrections, _model = val
        job.proofed.update(ids)
        for c in corrections:
            if not isinstance(c, dict) or not isinstance(c.get("i"), int) or c["i"] not in ids:
                stats["rejected"] += 1
                continue
            i, old, new = c["i"], c.get("old"), c.get("new")
            raw = job.ar[i - 1] or ""
            if not isinstance(old, str) or not isinstance(new, str) or raw.count(old) != 1 \
                    or not _correction_ok(old, new):
                stats["rejected"] += 1
                continue
            job.ar[i - 1] = raw.replace(old, new, 1)
            stats["applied"] += 1
            job.proof_log.append(f"الوحدة {i}: {old} -> {new}")

    run_parallel(lambda b: _proof_worker(job, chain, level, b), batches, min(PARALLEL[0], len(batches)), on_done)
    print(f"  proofreading: {stats['applied']} correction(s) applied, {stats['rejected']} rejected, "
          f"{stats['failed_batches']} batch(es) failed")


# ---------------------------------------------------------------- output
def autotag_terms(job, rendered, seen):
    """Glossary terms never marked by the model: add the English at their first appearance."""
    job.autotag_log = []
    entries = [(en, ar.strip()) for k, (en, ar) in list(job.gloss.items())
               if k not in seen and LATIN_RE.search(en) and ARABIC_RE.search(ar)
               and len(ar.strip()) >= 4 and ar.strip().lower() != en.lower()]
    for en, ar in entries:
        en_re = re.compile(r"(?<![A-Za-z])" + re.escape(en) + r"(?![A-Za-z])", re.I)
        ar_re = re.compile(r"(?<![\u0600-\u06FF])[وفبلك]?" + re.escape(ar) + r"(?![\u0600-\u06FF])")
        for i, u in enumerate(job.units):
            if job.kept_en[i] or job.ar[i] is None or not en_re.search(u["en"]):
                continue
            hit = None
            for m in ar_re.finditer(rendered[i]):
                if re.match(r"\s*\(\s*[A-Za-z]", rendered[i][m.end():m.end() + 40]):
                    continue
                hit = m
                break
            if hit:
                rendered[i] = rendered[i][:hit.end()] + f" ({en})" + rendered[i][hit.end():]
                seen.add(en.lower())
                job.autotag_log.append(f"الوحدة {i + 1}: {ar} ({en})")
                break


def render_all(job):
    seen, out = set(), []
    for i, u in enumerate(job.units):
        out.append(render_text(job.ar[i], seen) if job.ar[i] is not None else u["en"])
    autotag_terms(job, out, seen)
    return out


def write_output(path, job, rendered, bilingual):
    with open(path, "w", encoding="utf-8") as f:
        for u, ar in zip(job.units, rendered):
            if job.kind == "srt":
                f.write(f"{u['idx']}\n")
                if u["time"]:
                    f.write(f"{u['time']}\n")
                if bilingual:
                    f.write(f"{u['en']}\n")
                f.write(f"{ar}\n\n")
            else:
                if bilingual:
                    f.write(f"{u['en']}\n")
                f.write(f"{ar}\n\n")


_REASON_AR = [
    ("not translated", "لم تُترجم"),
    ("could not be translated", "تعذّرت ترجمتها"),
    ("answered by a fallback (lite) model", "ترجمها نموذج احتياطي (lite)"),
    ("translated by Gemma (mid tier): check terminology, meaning, omissions and grammar",
     "ترجمها Gemma (مستوى متوسط): تحقّق من المصطلحات والمعنى والحذف والقواعد"),
    ("possible omission (Arabic/English length", "احتمال حذف (نسبة طول العربي/الإنجليزي"),
    ("possible addition (Arabic/English length", "احتمال إضافة (نسبة طول العربي/الإنجليزي"),
    ("untranslated English: ", "إنجليزية غير مترجمة: "),
    ("numbers not found in Arabic: ", "أرقام غير موجودة في الترجمة العربية: "),
    ("glossary terms not marked: ", "مصطلحات من القاموس غير موسومة: "),
]
_STAGE_AR = {
    "context pass": "مرحلة السياق",
    "translation (quality models only)": "الترجمة (النماذج عالية الجودة فقط)",
    "translation (lite last resort)": "الترجمة (نموذج lite كحل أخير)",
    "review": "المراجعة",
    "proofreading": "التدقيق اللغوي",
}


def reason_ar(r):
    for en, ar in _REASON_AR:
        if r.startswith(en):
            return ar + r[len(en):]
    return r


def write_report(path, job, started):
    lines = [f"تقرير الترجمة ({time.strftime('%Y-%m-%d %H:%M:%S')}، {(time.time() - started) / 60:.1f} دقيقة)",
             f"نوع المدخل: {job.kind} | عدد الوحدات: {len(job.units)} | المجال: {job.brief.get('domain', '')}",
             "الطلبات لكل نموذج: " + ("، ".join(f"{m}: {c}" for m, c in USED.most_common()) or "-"),
             f"وحدات ما زالت مترجمة بنموذج احتياطي (lite): {sum(job.deg)}",
             f"وحدات ترجمها Gemma ولم تُراجَع بعد: {sum(job.gem)}",
             f"وحدات بقيت بالإنجليزية (فشلت ترجمتها): {sum(job.kept_en)}",
             f"وحدات تمت مراجعتها: {len(job.reviewed)} "
             f"(المعدَّلة منها: {sum(1 for v in job.reviewed.values() if v == 'changed')})",
             "", "== تصحيحات التعرّف على الكلام التي رصدتها مرحلة السياق =="]
    lines += [f"- {c.get('heard')} -> {c.get('meant')}" for c in job.brief.get("asr_corrections", [])] or ["(لا يوجد)"]
    lines += ["", "== ملاحظات متبقية بعد المراجعة =="]
    flagged = [(u, check_unit(job, u)) for u in job.units if job.ar[u["id"] - 1] is not None]
    flagged = [(u, r) for u, r in flagged if r]
    for u, r in flagged:
        lines.append(f"- الوحدة {u['id']} (تبدأ بـ: {u['en'][:70]!r}): {'؛ '.join(reason_ar(x) for x in r)}")
    if not flagged:
        lines.append("(لا يوجد)")
    lines += ["", "== سجل المراجعة =="] + (job.review_log or ["(لا يوجد)"])
    lines += ["", "== التدقيق اللغوي ==",
              f"التصحيحات المطبّقة: {job.proof_stats['applied']} | المرفوضة: {job.proof_stats['rejected']} "
              f"| الدفعات الفاشلة: {job.proof_stats['failed_batches']}"] + job.proof_log[:300]
    lines += ["", f"== مصطلحات وُسمت تلقائيًا (أضاف البرنامج الإنجليزية): {len(job.autotag_log)} =="] \
        + (job.autotag_log or ["(لا يوجد)"])
    lines += ["", "== التوقيت =="] + [f"{_STAGE_AR.get(n, n)}: {d / 60:.1f} دقيقة، {r} طلب" for n, d, r in TIMINGS]
    gused = {m: c for m, c in USED.items() if is_gemma(m)}
    lines += [f"مستوى Gemma: الطلبات {dict(gused) or 0} | وحدات أُبقي عليها من Gemma بعد المراجعة: "
              f"{sum(1 for k, m in enumerate(job.model) if is_gemma(m) and not job.gem[k])} "
              f"| أخطاء 503 لكل نموذج: " + ("، ".join(f"{m}: {n}" for (m, c), n in sorted(ERRS.items()) if c == 503)
                                          or "لا يوجد"),
              "بدائل قدرات النماذج: " + ("؛ ".join(QUIRK_LOG) or "لا يوجد")]
    lines += [f"الوقت المقضي في الانتظار/التأجيل (مجموع الخيوط): {WAIT[0] / 60:.1f} دقيقة",
              "أخطاء HTTP لكل نموذج: " + ("، ".join(f"{m} HTTP {c}: {n}" for (m, c), n in sorted(ERRS.items()))
                                         or "لا يوجد")]
    redo = [u["id"] for u in job.units if job.ar[u["id"] - 1] is None or job.deg[u["id"] - 1]
            or job.gem[u["id"] - 1] or job.kept_en[u["id"] - 1]]
    lines += ["", f"== وحدات تُعاد في التشغيل القادم (lite / إنجليزية / غير مترجمة): {len(redo)} ==", str(redo)]
    lines += ["", f"== القاموس النهائي ({len(job.gloss)} مصطلح) =="]
    lines += [f"{en} = {ar}" for en, ar in job.gloss.values()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_progress(path, job):
    with job.lock:
        data = {
            "fingerprint": job.fingerprint(), "brief": job.brief, "gloss": dict(job.gloss),
            "core": sorted(job.core), "ar": list(job.ar), "deg": list(job.deg), "gem": list(job.gem),
            "model": list(job.model),
            "kept_en": list(job.kept_en), "reviewed": {str(k): v for k, v in job.reviewed.items()},
            "proofed": sorted(job.proofed),
        }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)


def load_progress(path, job):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("fingerprint") != job.fingerprint() or len(data["ar"]) != len(job.units):
            print("Checkpoint found but it does not match this transcript -> ignored.")
            return False
        job.brief = data["brief"]
        job.brief.setdefault("keep_english", [])
        job.gloss = {k: tuple(v) for k, v in data["gloss"].items()}
        job.core = set(data["core"])
        job.ar, job.deg = data["ar"], data["deg"]
        job.model, job.kept_en = data["model"], data["kept_en"]
        job.gem = data.get("gem") or [False] * len(job.units)
        job.carry = {k for k, g in enumerate(job.gem) if g}
        job.reviewed = {int(k): v for k, v in data.get("reviewed", {}).items()}
        job.proofed = set(data.get("proofed", []))
        done = sum(1 for a in job.ar if a is not None)
        print(f"Resuming from checkpoint: {done}/{len(job.units)} units already translated.")
        return True
    except Exception as e:
        print(f"Checkpoint unreadable ({e!r}) -> ignored.")
        return False


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    started = time.time()

    if len(sys.argv) not in (3, 4, 5):
        print("Usage: python translate_transcript.py <input> <output_prefix> [arabic|bilingual|both] [model]")
        sys.exit(1)

    input_path, prefix = sys.argv[1], sys.argv[2]
    mode = sys.argv[3].strip().lower() if len(sys.argv) >= 4 else "arabic"
    model = sys.argv[4].strip() if len(sys.argv) == 5 and sys.argv[4].strip() else DEFAULT_MODEL
    if mode not in ("arabic", "bilingual", "both"):
        sys.exit("ERROR: mode must be arabic, bilingual or both.")

    KEYS[:] = load_keys()
    if not KEYS:
        sys.exit("ERROR: no Gemini API key found (set GEMINI_API_KEY as a GitHub secret).")

    try:
        minutes = float(os.environ.get("TRANSLATE_MAX_MINUTES", DEFAULT_MAX_MINUTES))
    except ValueError:
        minutes = DEFAULT_MAX_MINUTES
    START[0] = time.time()
    DEADLINE[0] = START[0] + minutes * 60

    ext = os.path.splitext(input_path)[1] or ".txt"
    segments = parse_transcript(input_path)
    if not segments:
        sys.exit("ERROR: no segments found in input.")
    kind = "srt" if any(s["time"] for s in segments) else "txt"
    units = build_units(segments, kind)
    job = Job(units, kind, model)
    for u in units:  # empty cues need no request
        if not u["en"].strip():
            job.ar[u["id"] - 1] = ""
    words = sum(u["words"] for u in units)
    try:
        PARALLEL[0] = max(1, min(8, int(os.environ.get("TRANSLATE_PARALLEL", min(3, len(KEYS))))))
    except ValueError:
        PARALLEL[0] = max(1, min(3, len(KEYS)))
    print(f"Parsed {len(segments)} segments -> {len(units)} {'cues' if kind == 'srt' else 'paragraphs'} "
          f"({words} words). Keys: {len(KEYS)} | quality chain: {' -> '.join(build_chain(model))} "
          f"| gemma: {' -> '.join(gemma_chain()) if gemma_allowed() else 'off'} "
          f"| last resort: {' -> '.join(last_resort_chain())} | thinking: {get_thinking_level() or 'default'} "
          f"| budget: {minutes:.0f} min | parallel: {PARALLEL[0]}")

    GATE.setup(PARALLEL[0])
    progress_path = f"{prefix}_progress.json"
    PROGRESS_PATH[0] = progress_path
    resumed = load_progress(progress_path, job)

    try:
        if not (resumed and job.brief.get("domain")):
            with Stage("context pass"):
                set_stage_end("context")
                print("Building context brief + glossary + mishearing list...")
                job.brief = build_brief(job)
                job.add_brief_glossary()
        print(f"  domain: {job.brief.get('domain')} | glossary terms: {len(job.gloss)} "
              f"| mishearings noted: {len(job.brief.get('asr_corrections', []))} "
              f"| keep-in-English: {len(job.brief.get('keep_english', []))}")
        save_progress(progress_path, job)

        def run_translation(label, redo_lite, on_fail_continue):
            rnd = 0
            while True:
                chunks = plan_chunks(job, redo_lite=redo_lite)
                if not chunks:
                    return
                if time.time() >= STAGE_END[0]:
                    print(f"  {label}: stage time is up with {sum(len(c) for c in chunks)} unit(s) pending")
                    return
                if not job.lite_ok and not chain_alive(build_chain(model)):
                    print("  every quality model has used its daily quota on every key "
                          "(free-tier quotas reset at midnight Pacific time) -> stopping this stage")
                    return
                rnd += 1
                items = list(enumerate(chunks, 1))
                print(f"{label} round {rnd}: {len(items)} batch(es), {sum(len(c) for c in chunks)} unit(s) pending")

                def tr(item):
                    n, chunk = item
                    print(f"Translating batch {n}/{len(items)} (units {chunk[0]['id']}-{chunk[-1]['id']} "
                          f"/ {len(units)})...")
                    translate_batch(job, chunk)

                def tr_done(item, ok, val):
                    if ok:
                        save_progress(progress_path, job)
                        return
                    if isinstance(val, StageTimeUp):
                        return
                    if isinstance(val, ApiUnavailable) and not val.fatal:
                        print(f"  batch {item[0]} not finished: {str(val)[:100]}")
                        save_progress(progress_path, job)
                        return
                    raise val

                run_parallel(tr, items, min(PARALLEL[0], len(items)), tr_done)
                if not plan_chunks(job, redo_lite=redo_lite) or on_fail_continue:
                    return
                left = STAGE_END[0] - time.time()
                if left <= 1:
                    continue
                wait = min(random.uniform(*RETRY_PAUSE), left)
                print(f"  quality models are busy; pausing {wait:.0f}s, then retrying the unfinished batches")
                with LOCK:
                    PAUSE_UNTIL[0] = max(PAUSE_UNTIL[0], time.time() + wait)
                try:
                    _wait_pause(STAGE_END[0])
                except ApiUnavailable:
                    pass

        with Stage("translation (quality models only)"):
            set_stage_end("translate")
            run_translation("Translation", True, False)

        if plan_chunks(job, redo_lite=False):
            with Stage("translation (lite last resort)"):
                set_stage_end("lite")
                job.lite_ok = True
                QUALITY_FAILS[0] = 2
                run_translation("Last-resort", False, True)
            job.lite_ok = False

        with Stage("review"):
            set_stage_end("review")
            review_pass(job)
        with Stage("proofreading"):
            set_stage_end("proof")
            proofread_pass(job)
    except ApiUnavailable as e:
        done = sum(1 for a in job.ar if a is not None)
        print(f"ERROR: {e}")
        save_progress(progress_path, job)
        p = f"{prefix}_ar_partial{ext}"
        write_output(p, job, render_all(job), bilingual=False)
        write_report(f"{prefix}_report.txt", job, started)
        print(f"Saved partial result ({done}/{len(units)} units translated, rest kept in English): {p}")
        print(f"Checkpoint saved: {progress_path} (commit it to the repo root to resume without repeating work)")
        sys.exit(1)

    rendered = render_all(job)
    print(f"Auto-tagged {len(job.autotag_log)} term(s) that the model had not marked.")
    if mode in ("arabic", "both"):
        p = f"{prefix}_ar{ext}"
        write_output(p, job, rendered, bilingual=False)
        print(f"Wrote {p}")
    if mode in ("bilingual", "both"):
        p = f"{prefix}_bilingual{ext}"
        write_output(p, job, rendered, bilingual=True)
        print(f"Wrote {p}")
    write_report(f"{prefix}_report.txt", job, started)
    print(f"Wrote {prefix}_report.txt")
    incomplete = any(a is None for a in job.ar) or any(job.deg) or any(job.gem) or any(job.kept_en)
    if incomplete:
        save_progress(progress_path, job)
        n = sum(1 for k in range(len(units)) if job.ar[k] is None or job.deg[k] or job.gem[k] or job.kept_en[k])
        print(f"INCOMPLETE: {n} unit(s) are untranslated, came from a lite model or are unreviewed Gemma output "
              f"(listed in the report). "
              f"Checkpoint kept: {progress_path} - commit it and run again after the daily quota resets "
              f"(midnight Pacific time); only those units will be redone.")
    else:
        try:
            os.remove(progress_path)
        except OSError:
            pass

    summary = ", ".join(f"{m}: {c}" for m, c in USED.most_common())
    print(f"Requests per model -> {summary}")
    print("Done.")


if __name__ == "__main__":
    main()
