"""
Professional, context-aware English -> Arabic translation of a transcript
(.txt numbered segments, or .srt) using the Gemini API.

Usage:
    python translate_transcript.py <input_path> <output_prefix> [mode] [model]

    mode:  "arabic" (default) | "bilingual" | "both"
    model: Gemini model name (default: gemini-3.8-flash)

Outputs (extension follows the input file):
    <output_prefix>_ar.<ext>          Arabic only
    <output_prefix>_bilingual.<ext>   English line + Arabic line

Requires env var GEMINI_API_KEY. Uses only the Python standard library.
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

CHUNK_SIZE = 50          # segments per request
CONTEXT_BEFORE = 6       # previous segments (EN + AR) given as context
CONTEXT_AFTER = 3        # upcoming EN segments given as lookahead
MAX_RETRIES = 6
DEFAULT_MODEL = "gemini-3.8-flash"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}")


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
def call_gemini(model, api_key, system, user, schema=None, temperature=0.3):
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
        },
    }
    if schema:
        body["generationConfig"]["responseSchema"] = schema
    data = json.dumps(body).encode("utf-8")
    url = API_URL.format(model=model)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            parts = out["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
            return text
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "ignore")[:300]
            last_err = f"HTTP {e.code}: {msg}"
            if e.code in (400, 401, 403, 404):
                raise SystemExit(f"ERROR: {last_err}")
        except Exception as e:  # network, malformed response, etc.
            last_err = repr(e)
        wait = min(2 ** attempt, 60)
        print(f"  API retry {attempt}/{MAX_RETRIES} in {wait}s ({last_err})")
        time.sleep(wait)
    raise SystemExit(f"ERROR: Gemini API failed after {MAX_RETRIES} retries: {last_err}")


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


def build_brief(model, api_key, segments):
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
        return clean_json(call_gemini(model, api_key, CONTEXT_SYSTEM, prompt, CONTEXT_SCHEMA, 0.2))
    except Exception as e:
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


def translate_chunk(model, api_key, brief, segments, translations, start, end, depth=0):
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
            result = clean_json(call_gemini(model, api_key, TRANSLATE_SYSTEM, prompt, CHUNK_SCHEMA))
            if isinstance(result, list) and len(result) == n and all(isinstance(x, str) for x in result):
                return [x.strip() for x in result]
            got = len(result) if isinstance(result, list) else "?"
            print(f"  count mismatch (expected {n}, got {got}), attempt {attempt}/3")
        except SystemExit:
            raise
        except Exception as e:
            print(f"  parse error attempt {attempt}/3: {e!r}")

    if n == 1:
        print(f"  WARNING: segment {segments[start]['idx']} could not be translated; keeping English.")
        return [segments[start]["text"]]
    mid = start + n // 2
    print(f"  splitting chunk {start+1}-{end} into halves")
    left = translate_chunk(model, api_key, brief, segments, translations, start, mid, depth + 1)
    for k, t in enumerate(left):
        translations[start + k] = t
    right = translate_chunk(model, api_key, brief, segments, translations, mid, end, depth + 1)
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

    ext = os.path.splitext(input_path)[1] or ".txt"
    segments = parse_transcript(input_path)
    if not segments:
        sys.exit("ERROR: no segments found in input.")
    print(f"Parsed {len(segments)} segments. Model: {model}")

    print("Building context brief + glossary...")
    brief = build_brief(model, api_key, segments)
    print(f"  domain: {brief.get('domain')} | glossary terms: {len(brief.get('glossary', []))}")

    translations = [None] * len(segments)
    for start in range(0, len(segments), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(segments))
        print(f"Translating segments {start+1}-{end} / {len(segments)}...")
        out = translate_chunk(model, api_key, brief, segments, translations, start, end)
        translations[start:end] = out

    if mode in ("arabic", "both"):
        p = f"{prefix}_ar{ext}"
        write_output(p, segments, translations, bilingual=False)
        print(f"Wrote {p}")
    if mode in ("bilingual", "both"):
        p = f"{prefix}_bilingual{ext}"
        write_output(p, segments, translations, bilingual=True)
        print(f"Wrote {p}")
    print("Done.")


if __name__ == "__main__":
    main()
