"""
Professional, context-aware English -> Arabic translation of a transcript
(.txt numbered segments, or .srt) using the Gemini API.

Usage:
    python translate_transcript.py <input_path> <output_prefix> [mode] [model]

    mode:  "arabic" (default) | "bilingual" | "both"
    model: primary Gemini model name (default: gemini-3.8-flash)

Resilience (HTTP 503 / overload):
    Every request walks a fallback chain: primary model -> FALLBACK_CHAIN.
    Each model gets a few jittered-backoff retries, then the next model is
    tried. A model that just failed is skipped for COOLDOWN_SECS so later
    requests do not waste time on it. If the whole chain fails, a partial
    file (<prefix>_ar_partial.<ext>) is written and the script exits 1.

Optional env vars:
    GEMINI_API_KEY          (required)
    GEMINI_FALLBACK_MODELS  comma-separated override of the fallback chain
    GEMINI_THINKING_LEVEL   low (default) | medium | high | off

Outputs (extension follows the input file):
    <output_prefix>_ar.<ext>          Arabic only
    <output_prefix>_bilingual.<ext>   English line + Arabic line

Uses only the Python standard library.
"""

import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter

CHUNK_SIZE = 50          # segments per request
CONTEXT_BEFORE = 6       # previous segments (EN + AR) given as context
CONTEXT_AFTER = 3        # upcoming EN segments given as lookahead

DEFAULT_MODEL = "gemini-3.8-flash"
# Ordered by quality; the primary model is always tried first.
FALLBACK_CHAIN = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

PER_MODEL_RETRIES = 3    # attempts per model before moving to the next one
CHAIN_ROUNDS = 2         # full passes over the chain before giving up
ROUND_WAIT = 45          # seconds between full passes
COOLDOWN_SECS = 90       # skip a just-failed model for this long
REQUEST_TIMEOUT = 240

RETRYABLE_HTTP = {429, 500, 502, 503, 504}
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}")

COOLDOWN = {}            # model -> unix time until which it is skipped
NO_THINKING = set()      # models that rejected thinkingConfig
USED = Counter()         # successful requests per model


class ApiUnavailable(Exception):
    """Every model in the chain failed (overload / network / server errors)."""


def get_thinking_level():
    lvl = os.environ.get("GEMINI_THINKING_LEVEL", "low").strip().lower()
    return None if lvl in ("", "off", "none") else lvl


def build_chain(primary):
    env = os.environ.get("GEMINI_FALLBACK_MODELS", "").strip()
    fallbacks = [m.strip() for m in env.split(",") if m.strip()] if env else FALLBACK_CHAIN
    chain = [primary]
    for m in fallbacks:
        if m not in chain:
            chain.append(m)
    return chain


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
    cand = out["candidates"][0]
    parts = cand["content"]["parts"]
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    if not text.strip():
        raise ValueError(f"empty response (finishReason={cand.get('finishReason')})")
    return text


def _sleep_backoff(attempt):
    wait = min(2 ** attempt, 30)
    wait += random.uniform(0, wait * 0.5)  # jitter
    time.sleep(wait)
    return wait


def call_gemini(chain, api_key, system, user, schema=None, temperature=0.3):
    """Try each model in `chain` (with retries) until one answers."""
    level = get_thinking_level()
    last_err = None

    for rnd in range(1, CHAIN_ROUNDS + 1):
        now = time.time()
        order = [m for m in chain if COOLDOWN.get(m, 0) <= now] or list(chain)

        for model in order:
            use_thinking = bool(level) and model not in NO_THINKING
            for attempt in range(1, PER_MODEL_RETRIES + 1):
                body = _build_body(system, user, schema, level if use_thinking else None)
                try:
                    text = _post(model, api_key, body)
                    USED[model] += 1
                    if model != chain[0]:
                        print(f"  [fallback] answered by {model}")
                    return text
                except urllib.error.HTTPError as e:
                    msg = e.read().decode("utf-8", "ignore")[:300]
                    last_err = f"{model} HTTP {e.code}: {msg}"
                    if e.code in (401, 403):
                        raise SystemExit(f"ERROR: {last_err}")
                    if e.code == 400 and use_thinking:
                        NO_THINKING.add(model)
                        use_thinking = False
                        print(f"  {model}: thinkingConfig rejected, retrying without it")
                        continue
                    if e.code not in RETRYABLE_HTTP:
                        print(f"  {model}: non-retryable HTTP {e.code}, skipping model")
                        break
                except Exception as e:  # network, timeout, malformed/empty response
                    last_err = f"{model} {e!r}"

                if attempt < PER_MODEL_RETRIES:
                    print(f"  {model} retry {attempt}/{PER_MODEL_RETRIES - 1} ({last_err[:120]})")
                    _sleep_backoff(attempt)

            COOLDOWN[model] = time.time() + COOLDOWN_SECS
            print(f"  {model} unavailable -> trying next model")

        if rnd < CHAIN_ROUNDS:
            wait = ROUND_WAIT + random.uniform(0, 15)
            print(f"  all models failed (round {rnd}/{CHAIN_ROUNDS}); waiting {wait:.0f}s")
            time.sleep(wait)
            COOLDOWN.clear()

    raise ApiUnavailable(f"all models failed after {CHAIN_ROUNDS} rounds: {last_err}")


def clean_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


# ---------------------------------------------------------------- context pass
CONTEXT_SYSTEM = (
    "You are a senior translator and subject-matter editor preparing an "
    "English-to-Arabic translation of a spoken transcript (podcast/lecture)."
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
    },
    "required": ["domain", "summary", "tone", "glossary"],
}


def build_brief(chain, api_key, segments):
    full = "\n".join(s["text"] for s in segments)[:400_000]
    prompt = (
        "Read this full transcript and return JSON with:\n"
        "- domain: the subject field (e.g. medicine/neuroanatomy).\n"
        "- summary: 3-4 sentences on what it covers and the speakers' dynamic.\n"
        "- tone: the register to use in Arabic (e.g. educational, conversational).\n"
        "- glossary: up to 80 key terms/recurring expressions with the single "
        "best standard Arabic rendering each (use the established Arabic "
        "scientific/medical term when one exists) so terminology stays consistent.\n\n"
        f"TRANSCRIPT:\n{full}"
    )
    try:
        return clean_json(call_gemini(chain, api_key, CONTEXT_SYSTEM, prompt, CONTEXT_SCHEMA, 0.2))
    except Exception as e:  # ApiUnavailable / bad JSON: continue without a brief
        print(f"  (context pass failed, continuing without brief: {e!r})")
        return {"domain": "", "summary": "", "tone": "", "glossary": []}


# ---------------------------------------------------------------- translation
TRANSLATE_SYSTEM = """You are a professional English-to-Arabic translator specialising in spoken educational content.
Rules:
1. Translate into clear, natural Modern Standard Arabic (فصحى مبسطة) that reads as if originally written in Arabic, not word-for-word. Keep the speakers' tone (curiosity, humour, emphasis) and conversational fillers only when they carry meaning.
2. Translate by MEANING and CONTEXT: segments are fragments of continuing sentences, so use the surrounding segments to resolve pronouns, ambiguity and idioms. Never translate idioms literally.
3. Use established Arabic scientific/medical terminology. On a term's first appearance, put the English term in parentheses after the Arabic; afterwards use Arabic only. Follow the provided glossary exactly for consistency.
4. Keep numbers, units, drug/anatomical names accurate. Keep proper nouns, brands and acronyms in Latin script when customary (e.g. CT, MRI).
5. Output EXACTLY one Arabic string per input segment, in the same order. Never merge, split, skip or add segments. A short interjection ("Yeah.", "Right.") gets a short natural Arabic equivalent.
6. Return only JSON: an array of strings."""

CHUNK_SCHEMA = {"type": "ARRAY", "items": {"type": "STRING"}}


def brief_text(brief):
    gl = "\n".join(f"- {g['en']} = {g['ar']}" for g in brief.get("glossary", []))
    return (
        f"Domain: {brief.get('domain','')}\nSummary: {brief.get('summary','')}\n"
        f"Arabic tone: {brief.get('tone','')}\nGlossary:\n{gl}"
    )


def translate_chunk(chain, api_key, brief, segments, translations, start, end, depth=0):
    """Translate segments[start:end]; returns list of Arabic strings (len == end-start)."""
    n = end - start
    before = range(max(0, start - CONTEXT_BEFORE), start)
    after = range(end, min(len(segments), end + CONTEXT_AFTER))

    ctx_prev = "\n".join(
        f"EN: {segments[i]['text']}\nAR: {translations[i]}" for i in before if translations[i] is not None
    )
    ctx_next = "\n".join(segments[i]["text"] for i in after)
    items = json.dumps([segments[i]["text"] for i in range(start, end)], ensure_ascii=False)

    prompt = (
        f"{brief_text(brief)}\n\n"
        f"PREVIOUS SEGMENTS (already translated, for continuity only):\n{ctx_prev or '(none)'}\n\n"
        f"UPCOMING SEGMENTS (context only, do NOT translate):\n{ctx_next or '(none)'}\n\n"
        f"TRANSLATE these {n} segments. Return a JSON array of exactly {n} Arabic strings:\n{items}"
    )

    for attempt in range(1, 4):
        try:
            result = clean_json(call_gemini(chain, api_key, TRANSLATE_SYSTEM, prompt, CHUNK_SCHEMA))
            if isinstance(result, list) and len(result) == n and all(isinstance(x, str) for x in result):
                return [x.strip() for x in result]
            got = len(result) if isinstance(result, list) else "?"
            print(f"  count mismatch (expected {n}, got {got}), attempt {attempt}/3")
        except (SystemExit, ApiUnavailable):
            raise
        except Exception as e:
            print(f"  parse error attempt {attempt}/3: {e!r}")

    if n == 1:
        print(f"  WARNING: segment {segments[start]['idx']} could not be translated; keeping English.")
        return [segments[start]["text"]]
    mid = start + n // 2
    print(f"  splitting chunk {start+1}-{end} into halves")
    left = translate_chunk(chain, api_key, brief, segments, translations, start, mid, depth + 1)
    for k, t in enumerate(left):
        translations[start + k] = t
    right = translate_chunk(chain, api_key, brief, segments, translations, mid, end, depth + 1)
    return left + right


# ---------------------------------------------------------------- output
def write_output(path, segments, translations, bilingual):
    with open(path, "w", encoding="utf-8") as f:
        for seg, ar in zip(segments, translations):
            f.write(f"{seg['idx']}\n")
            if seg["time"]:
                f.write(f"{seg['time']}\n")
            if bilingual:
                f.write(f"{seg['text']}\n")
            f.write(f"{ar}\n\n")


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if len(sys.argv) not in (3, 4, 5):
        print("Usage: python translate_transcript.py <input> <output_prefix> [arabic|bilingual|both] [model]")
        sys.exit(1)

    input_path, prefix = sys.argv[1], sys.argv[2]
    mode = sys.argv[3].strip().lower() if len(sys.argv) >= 4 else "arabic"
    model = sys.argv[4].strip() if len(sys.argv) == 5 and sys.argv[4].strip() else DEFAULT_MODEL
    if mode not in ("arabic", "bilingual", "both"):
        sys.exit("ERROR: mode must be arabic, bilingual or both.")

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        sys.exit("ERROR: GEMINI_API_KEY environment variable is not set (add it as a GitHub secret).")

    chain = build_chain(model)
    ext = os.path.splitext(input_path)[1] or ".txt"
    segments = parse_transcript(input_path)
    if not segments:
        sys.exit("ERROR: no segments found in input.")
    print(f"Parsed {len(segments)} segments. Model chain: {' -> '.join(chain)}")

    translations = [None] * len(segments)
    try:
        print("Building context brief + glossary...")
        brief = build_brief(chain, api_key, segments)
        print(f"  domain: {brief.get('domain')} | glossary terms: {len(brief.get('glossary', []))}")

        for start in range(0, len(segments), CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, len(segments))
            print(f"Translating segments {start+1}-{end} / {len(segments)}...")
            out = translate_chunk(chain, api_key, brief, segments, translations, start, end)
            translations[start:end] = out
    except ApiUnavailable as e:
        done = sum(1 for t in translations if t is not None)
        print(f"ERROR: {e}")
        partial = [t if t is not None else s["text"] for s, t in zip(segments, translations)]
        p = f"{prefix}_ar_partial{ext}"
        write_output(p, segments, partial, bilingual=False)
        print(f"Saved partial result ({done}/{len(segments)} translated, rest kept in English): {p}")
        sys.exit(1)

    if mode in ("arabic", "both"):
        p = f"{prefix}_ar{ext}"
        write_output(p, segments, translations, bilingual=False)
        print(f"Wrote {p}")
    if mode in ("bilingual", "both"):
        p = f"{prefix}_bilingual{ext}"
        write_output(p, segments, translations, bilingual=True)
        print(f"Wrote {p}")

    summary = ", ".join(f"{m}: {c}" for m, c in USED.most_common())
    print(f"Requests per model -> {summary}")
    print("Done.")


if __name__ == "__main__":
    main()
