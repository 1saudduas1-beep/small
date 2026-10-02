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
    TRANSLATE_PARALLEL       parallel batches (default: min(4, number of keys))
    TRANSLATE_PROOFREAD      on (default) | off
    TRANSLATE_PROOF_BATCH    paragraphs per proofreading request (default 25)
    TRANSLATE_PROOF_LEVEL    thinking level of the proofreading pass (default medium)
    TRANSLATE_MAX_MINUTES    global time budget (default 120)
    TRANSLATE_REVIEW_MODEL   default gemini-3.1-pro-preview ; "off" disables review
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
CHUNK_WORDS = 1200           # .txt: English words per request
CHUNK_MAX_UNITS = 12
SRT_CHUNK_CUES = 40
CTX = {"txt": (2, 1), "srt": (6, 3)}   # (units before, units after) given as context

DEFAULT_MODEL = "gemini-3.8-flash"
QUALITY_CHAIN = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.7-flash"]
LAST_RESORT = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
DEFAULT_REVIEW_MODEL = "gemini-3.1-pro-preview"

PER_MODEL_RETRIES = 4
CHAIN_ROUNDS = 2
ROUND_WAIT = 45
COOLDOWN_SECS = 60
MAX_SERVER_DELAY = 90
REQUEST_TIMEOUT = 300
DEFAULT_MAX_MINUTES = 120
QUALITY_PATIENCE = 240       # seconds spent on the quality chain per request
SHORT_PATIENCE = 45          # after 2 consecutive degradations
REVIEW_BATCH = 6
OVERLOAD_HTTP = {503}        # model overloaded: fail over at once, never rotate keys
DEMOTE_AFTER = 2             # consecutive failures before a model is demoted
DEMOTE_SECS = 600            # a demoted model is tried last for this long
PROOF_BATCH = 25
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
WAIT = [0.0]             # seconds spent sleeping (summed over threads)
TIMINGS = []             # (stage, seconds, requests)


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


def note_failure(model):
    with LOCK:
        MODEL_FAILS[model] += 1
        if MODEL_FAILS[model] >= DEMOTE_AFTER:
            DEMOTED[model] = time.time() + DEMOTE_SECS
            if MODEL_FAILS[model] == DEMOTE_AFTER:
                print(f"  {model}: {DEMOTE_AFTER} failures in a row -> demoted for {DEMOTE_SECS // 60} min")


def note_success(model):
    with LOCK:
        USED[model] += 1
        MODEL_FAILS[model] = 0
        DEMOTED.pop(model, None)


class ApiUnavailable(Exception):
    def __init__(self, msg, fatal=False):
        super().__init__(msg)
        self.fatal = fatal


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
def _build_body(system, user, schema, thinking):
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    if schema:
        body["generationConfig"]["responseSchema"] = schema
    if thinking:
        body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": thinking}
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


def call_gemini(chain, system, user, schema=None, level="default", patience=None):
    """Walk `chain` (models) x keys until one answers. Returns (text, model)."""
    if level == "default":
        level = get_thinking_level()
    stop_at = min(DEADLINE[0], time.time() + patience) if patience else DEADLINE[0]
    last_err = None

    for rnd in range(1, CHAIN_ROUNDS + 1):
        live = [m for m in chain if m not in DEAD_MODELS
                and any(_key_ok(k, m) for k in range(len(KEYS)))]
        if not live:
            raise ApiUnavailable(f"no usable model/key left ({last_err})", fatal=not _any_key())
        now = time.time()
        with LOCK:
            ready = [m for m in live if MODEL_COOL.get(m, 0) <= now and DEMOTED.get(m, 0) <= now]
        order = ready + [m for m in live if m not in ready]   # healthy models first

        for model in order:
            use_thinking = bool(level) and model not in NO_THINKING
            overloaded = False
            for attempt in range(1, PER_MODEL_RETRIES + 1):
                _guard(stop_at)
                ki = _pick_key(model, stop_at)
                if ki is None:
                    break
                body = _build_body(system, user, schema, level if use_thinking else None)
                try:
                    text = _post(model, KEYS[ki], body)
                    note_success(model)
                    return text, model
                except ResponseRejected:
                    raise
                except urllib.error.HTTPError as e:
                    raw = e.read().decode("utf-8", "ignore")
                    last_err = f"{model} (key #{ki + 1}) HTTP {e.code}: {raw[:160]}"
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
                    if e.code == 400 and use_thinking:
                        NO_THINKING.add(model)
                        use_thinking = False
                        print(f"  {model}: thinkingConfig rejected, retrying without it")
                        continue
                    if e.code in OVERLOAD_HTTP:
                        note_failure(model)
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

                if attempt < PER_MODEL_RETRIES:
                    print(f"  {model} retry {attempt}/{PER_MODEL_RETRIES - 1} ({last_err[:110]})")
                    _backoff(attempt, stop_at)

            MODEL_COOL[model] = time.time() + COOLDOWN_SECS
            if not overloaded:
                note_failure(model)
                print(f"  {model} unavailable -> trying next model")

        if rnd < CHAIN_ROUNDS:
            wait = ROUND_WAIT + random.uniform(0, 15)
            print(f"  all models failed (round {rnd}/{CHAIN_ROUNDS}); waiting {wait:.0f}s")
            _sleep(wait, stop_at)
            MODEL_COOL.clear()

    raise ApiUnavailable(f"all models failed: {last_err}")


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
    return json.loads(text)


# ------------------------------------------------------------ markup helpers
def strip_markup(text):
    text = MARK_RE.sub(lambda m: m.group(1), text)
    text = STRAY_RE.sub(lambda m: m.group(1).split("|")[0], text)
    return text.replace("⟦", "").replace("⟧", "")


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
        self.model = [""] * n
        self.kept_en = [False] * n
        self.reviewed = {}            # id -> "changed" | "unchanged"
        self.brief = {"domain": "", "summary": "", "tone": "", "glossary": [], "asr_corrections": [],
                      "keep_english": []}
        self.lock = threading.RLock()
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


def glossary_text(job):
    with job.lock:
        pairs = list(job.gloss.values())
    return "\n".join(f"- {en} -> {ar}" for en, ar in pairs) or "(none)"


def keep_text(job):
    return ", ".join(str(x) for x in job.brief.get("keep_english", []) if x) or "(none)"


def corrections_text(job):
    rows = [f"- {c.get('heard', '')} -> {c.get('meant', '')}" for c in job.brief.get("asr_corrections", [])
            if c.get("heard") and c.get("meant")]
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


def translate_batch(job, batch):
    """Translate `batch` (list of units). Re-requests only what is missing; splits on repeated failure."""
    system = SYSTEM_SRT if job.kind == "srt" else SYSTEM_TXT
    nb, na = CTX[job.kind]
    pending = list(batch)

    for attempt in range(1, 4):
        a, b = pending[0]["id"] - 1, pending[-1]["id"] - 1
        arabic_ctx = PARALLEL[0] <= 1
        ctx_prev = context_lines(job, range(max(0, a - nb), a), arabic_ctx)
        prev_label = ("PREVIOUS ITEMS (already translated, for continuity only)" if arabic_ctx else
                      "PREVIOUS ITEMS (English source, context only - do NOT translate; "
                      "stay consistent through the GLOSSARY)")
        ctx_next = context_lines(job, range(b + 1, min(len(job.units), b + 1 + na)), False)
        items = json.dumps([{"i": u["id"], "en": u["en"]} for u in pending], ensure_ascii=False)
        prompt = (
            f"{brief_text(job)}\n\n"
            f"GLOSSARY (use these Arabic renderings exactly):\n{glossary_text(job)}\n\n"
            f"KNOWN TRANSCRIPT MISHEARINGS (heard -> meant):\n{corrections_text(job)}\n\n"
            f"KEEP IN ENGLISH (leave exactly as written):\n{keep_text(job)}\n\n"
            f"{prev_label}:\n{ctx_prev or '(none)'}\n\n"
            f"UPCOMING ITEMS (context only, do NOT translate):\n{ctx_next or '(none)'}\n\n"
            f"TRANSLATE these {len(pending)} items. Return a JSON array of exactly {len(pending)} objects "
            f'{{"i", "ar"}} using the same i values:\nITEMS:\n{items}'
        )
        try:
            text, model, degraded = call_tiered(job.primary, system, prompt, CHUNK_SCHEMA)
            result = clean_json(text)
        except (SystemExit, ApiUnavailable):
            raise
        except ResponseRejected as e:
            print(f"  response rejected: {e}")
            break  # retrying the same batch will not help; split it
        except Exception as e:
            print(f"  parse error attempt {attempt}/3: {e!r}")
            continue

        got, problems = parse_items(result, pending)
        for i, ar in got.items():
            job.ar[i - 1] = ar
            job.deg[i - 1] = degraded
            job.model[i - 1] = model
            job.learn_terms(ar)
        pending = [u for u in pending if u["id"] not in got]
        if not pending:
            return
        print(f"  invalid/missing: {'; '.join(problems[:3])} (attempt {attempt}/3)")

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


def plan_chunks(job):
    chunks, cur, words = [], [], 0
    limit_units = SRT_CHUNK_CUES if job.kind == "srt" else CHUNK_MAX_UNITS
    for u in job.units:
        if job.ar[u["id"] - 1] is not None:
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
    raw = job.ar[i] or ""
    plain = strip_markup(raw)
    en = u["en"]
    reasons = []
    if job.kept_en[i]:
        return ["could not be translated"]
    if job.deg[i]:
        reasons.append("answered by a fallback (lite) model")
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
        job.review_log.append("review disabled (TRANSLATE_REVIEW_MODEL=off)")
        return
    try:
        cap = int(os.environ.get("TRANSLATE_REVIEW_MAX", "15"))
    except ValueError:
        cap = 15
    suspects = []
    for u in job.units:
        reasons = check_unit(job, u)
        if reasons:
            suspects.append((u, reasons))
    if not suspects:
        print("Checks: no suspect units.")
        return
    must = [t for t in suspects if job.deg[t[0]["id"] - 1] or job.kept_en[t[0]["id"] - 1]]
    others = sorted((t for t in suspects if t not in must), key=lambda t: -len(t[1]))
    chosen = must + others[:cap]
    print(f"Checks: {len(suspects)} suspect unit(s); reviewing {len(chosen)} "
          f"({len(must)} lite/failed always, up to {cap} others).")
    if len(others) > cap:
        job.review_log.append(f"{len(others) - cap} lower-priority suspect unit(s) not reviewed (cap {cap})")
    chain = [rm] + [m for m in build_chain(job.primary) if m != rm]
    parts = [chosen[k:k + REVIEW_BATCH] for k in range(0, len(chosen), REVIEW_BATCH)]
    stopped = []

    def on_done(part, ok, val):
        ids = [u["id"] for u, _ in part]
        if not ok:
            reason = str(val)[:140]
            job.review_log.append(f"units {ids}: review NOT done ({reason})")
            print(f"  review batch {ids} skipped: {reason}")
            if isinstance(val, ApiUnavailable) and val.fatal:
                stopped.append(reason)
            return
        got, problems, model = val
        for i, ar in got.items():
            old = job.ar[i - 1]
            changed = strip_markup(ar) != strip_markup(old)
            job.ar[i - 1] = ar
            job.deg[i - 1] = False                       # a non-lite model has now read it
            if ARABIC_RE.search(strip_markup(ar)):
                job.kept_en[i - 1] = False
            job.reviewed[i] = "changed" if changed else "unchanged"
            job.learn_terms(ar)
            job.review_log.append(f"unit {i}: {'revised' if changed else 'confirmed'} by {model}")
        missing = [i for i in ids if i not in got]
        if missing:
            job.review_log.append(f"units {missing}: reviewer did not return them ({'; '.join(problems[:2])})")
            print(f"  review: {'; '.join(problems[:2])}")

    run_parallel(lambda part: _review_worker(job, chain, part), parts, min(PARALLEL[0], len(parts)), on_done)
    if stopped:
        job.review_log.append(f"REVIEW STOPPED EARLY: {stopped[0]}")
    left = [u["id"] for u in job.units if job.deg[u["id"] - 1]]
    if left:
        job.review_log.append(f"units still answered by a lite model after review: {left}")


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
    todo = [u for u in job.units if job.ar[u["id"] - 1] and not job.kept_en[u["id"] - 1]]
    batches = [todo[k:k + size] for k in range(0, len(todo), size)]
    chain = build_chain(job.primary)
    print(f"Proofreading {len(todo)} paragraph(s) in {len(batches)} request(s)...")
    stats = job.proof_stats

    def on_done(batch, ok, val):
        ids = [u["id"] for u in batch]
        if not ok:
            stats["failed_batches"] += 1
            job.proof_log.append(f"batch {ids[0]}-{ids[-1]}: proofreading NOT done ({str(val)[:140]})")
            print(f"  proofreading batch {ids[0]}-{ids[-1]} skipped: {str(val)[:100]}")
            return
        corrections, _model = val
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
            job.proof_log.append(f"unit {i}: {old} -> {new}")

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
                job.autotag_log.append(f"unit {i + 1}: {ar} ({en})")
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


def write_report(path, job, started):
    lines = [f"Translation report ({time.strftime('%Y-%m-%d %H:%M:%S')}, {(time.time() - started) / 60:.1f} min)",
             f"Input kind: {job.kind} | units: {len(job.units)} | domain: {job.brief.get('domain', '')}",
             "Requests per model: " + (", ".join(f"{m}: {c}" for m, c in USED.most_common()) or "-"),
             f"Units still marked as answered by a fallback (lite) model: {sum(job.deg)}",
             f"Units kept in English (failed): {sum(job.kept_en)}",
             f"Units reviewed: {len(job.reviewed)} "
             f"(revised: {sum(1 for v in job.reviewed.values() if v == 'changed')})",
             "", "== Speech-recognition corrections noted by the context pass =="]
    lines += [f"- {c.get('heard')} -> {c.get('meant')}" for c in job.brief.get("asr_corrections", [])] or ["(none)"]
    lines += ["", "== Remaining flags after review =="]
    flagged = [(u, check_unit(job, u)) for u in job.units if job.ar[u["id"] - 1] is not None]
    flagged = [(u, r) for u, r in flagged if r]
    for u, r in flagged:
        lines.append(f"- unit {u['id']} (starts: {u['en'][:70]!r}): {'; '.join(r)}")
    if not flagged:
        lines.append("(none)")
    lines += ["", "== Review log =="] + (job.review_log or ["(none)"])
    lines += ["", "== Language proofreading ==",
              f"corrections applied: {job.proof_stats['applied']} | rejected: {job.proof_stats['rejected']} "
              f"| failed batches: {job.proof_stats['failed_batches']}"] + job.proof_log[:300]
    lines += ["", f"== Auto-tagged terms (English added by the program): {len(job.autotag_log)} =="] \
        + (job.autotag_log or ["(none)"])
    lines += ["", "== Timing =="] + [f"{n}: {d / 60:.1f} min, {r} request(s)" for n, d, r in TIMINGS]
    lines += [f"time spent waiting/back-off (summed over threads): {WAIT[0] / 60:.1f} min"]
    lines += ["", f"== Final glossary ({len(job.gloss)} terms) =="]
    lines += [f"{en} = {ar}" for en, ar in job.gloss.values()]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_progress(path, job):
    with job.lock:
        data = {
            "fingerprint": job.fingerprint(), "brief": job.brief, "gloss": dict(job.gloss),
            "core": sorted(job.core), "ar": list(job.ar), "deg": list(job.deg), "model": list(job.model),
            "kept_en": list(job.kept_en),
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
    DEADLINE[0] = time.time() + minutes * 60

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
        PARALLEL[0] = max(1, min(8, int(os.environ.get("TRANSLATE_PARALLEL", min(4, len(KEYS))))))
    except ValueError:
        PARALLEL[0] = max(1, min(4, len(KEYS)))
    print(f"Parsed {len(segments)} segments -> {len(units)} {'cues' if kind == 'srt' else 'paragraphs'} "
          f"({words} words). Keys: {len(KEYS)} | quality chain: {' -> '.join(build_chain(model))} "
          f"| last resort: {' -> '.join(last_resort_chain())} | thinking: {get_thinking_level() or 'default'} "
          f"| budget: {minutes:.0f} min | parallel: {PARALLEL[0]}")

    progress_path = f"{prefix}_progress.json"
    resumed = load_progress(progress_path, job)

    try:
        if not (resumed and job.brief.get("domain")):
            with Stage("context pass"):
                print("Building context brief + glossary + mishearing list...")
                job.brief = build_brief(job)
                job.add_brief_glossary()
        print(f"  domain: {job.brief.get('domain')} | glossary terms: {len(job.gloss)} "
              f"| mishearings noted: {len(job.brief.get('asr_corrections', []))} "
              f"| keep-in-English: {len(job.brief.get('keep_english', []))}")
        save_progress(progress_path, job)

        chunks = list(enumerate(plan_chunks(job), 1))
        total = len(chunks)

        def tr(item):
            n, chunk = item
            print(f"Translating batch {n}/{total} (units {chunk[0]['id']}-{chunk[-1]['id']} / {len(units)})...")
            translate_batch(job, chunk)

        def tr_done(item, ok, val):
            if not ok:
                raise val
            save_progress(progress_path, job)

        with Stage("translation"):
            run_parallel(tr, chunks, min(PARALLEL[0], max(1, total)), tr_done)
        with Stage("review"):
            review_pass(job)
        with Stage("proofreading"):
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
    try:
        os.remove(progress_path)
    except OSError:
        pass

    summary = ", ".join(f"{m}: {c}" for m, c in USED.most_common())
    print(f"Requests per model -> {summary}")
    print("Done.")


if __name__ == "__main__":
    main()
