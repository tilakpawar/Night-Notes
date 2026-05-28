#!/usr/bin/env python3
"""
Night Notes — Local Server
- Transcribes audio using faster-whisper (local, model cached at startup)
- Proxies to Ollama for title extraction + summarization
- Serves the frontend HTML

Performance optimisations:
  * WhisperModel loaded ONCE at startup — not on every request
  * ThreadingHTTPServer — transcription and titlize run concurrently
  * beam_size=2 — ~2x faster on CPU, negligible accuracy loss for speech
  * VAD padding tightened — fewer wasted frames around silence
  * Ollama warmed at startup — first real request hits a hot model
  * num_predict capped tightly per use-case

Transcription quality fixes:
  * initial_prompt — primes Whisper's vocabulary/style, reduces hallucinations
  * no_speech_threshold lowered — less aggressive silence dropping
  * repetition_penalty — penalises the word-doubling artefact
  * post_process_segments() — strips any surviving adjacent duplicates
"""

import os
import re
import json
import tempfile
import threading
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse
import subprocess

PORT         = 8765
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:14b"

# Whisper initial_prompt: sets tone, vocabulary and style.
# Mentioning common filler words + proper noun style reduces hallucinations.
WHISPER_PROMPT = (
    "This is a personal voice journal entry. "
    "The speaker may mix English and Hindi mid-sentence. "
    "Transcribe exactly what is said, including filler words like um, uh, like. "
    "Do not add punctuation that wasn't implied by the speaker's pauses."
)

# ---------------------------------------------------------------------------
# Whisper model — loaded ONCE at startup, reused for every transcription
# ---------------------------------------------------------------------------

_whisper_model = None
_whisper_lock  = threading.Lock()

def get_whisper_model():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            try:
                from faster_whisper import WhisperModel
                print("  Loading Whisper large-v3… (one-time, ~5 s)")
                _whisper_model = WhisperModel(
                    "large-v3",
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=os.cpu_count() or 4,
                    num_workers=1,
                )
                print("  ✓ Whisper ready")
            except ImportError:
                pass
    return _whisper_model


# ---------------------------------------------------------------------------
# Post-processing: remove adjacent duplicate words/phrases Whisper hallucinates
# ---------------------------------------------------------------------------

def clean_transcript(text: str) -> str:
    # 1. Collapse duplicate adjacent words (e.g. "the the", "dr Dr")
    #    case-insensitive, keeps the first occurrence
    text = re.sub(
        r'\b(\w+)\s+\1\b',
        r'\1',
        text,
        flags=re.IGNORECASE,
    )

    # 2. Collapse duplicate adjacent short phrases (2-4 words), e.g.
    #    "let's see let's see" → "let's see"
    text = re.sub(
        r'\b((?:\w+\s+){1,3}\w+)\s+\1\b',
        r'\1',
        text,
        flags=re.IGNORECASE,
    )

    # 3. Remove the hallucinated silence/music tokens Whisper sometimes emits
    text = re.sub(r'\[.*?\]', '', text)         # [Music], [Silence], etc.
    text = re.sub(r'\(.*?\)', '', text)         # (Music), (Applause), etc.

    # 4. Tidy up leftover whitespace
    text = re.sub(r'  +', ' ', text).strip()

    return text


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def transcribe_audio(audio_bytes: bytes, filename: str) -> dict:
    suffix = os.path.splitext(filename)[-1] or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = get_whisper_model()
        if model is not None:
            segments, info = model.transcribe(
                tmp_path,
                beam_size=2,                    # fast; 2 is plenty for speech
                best_of=1,
                language=None,                  # per-segment detection for code-switching
                task="transcribe",
                multilingual=True,
                initial_prompt=WHISPER_PROMPT,  # steers vocabulary + reduces hallucinations
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=200,  # don't drop short pauses mid-sentence
                    speech_pad_ms=100,
                    threshold=0.3,                # lower = keep more borderline speech
                ),
                no_speech_threshold=0.4,          # default 0.6 is too aggressive
                log_prob_threshold=-1.2,          # drop only very low-confidence segments
                repetition_penalty=1.3,           # penalise the word-doubling artefact
                condition_on_previous_text=True,  # keep context for name/word consistency
            )
            raw_text = " ".join(seg.text.strip() for seg in segments)
            text = clean_transcript(raw_text)
            return {"text": text, "language": info.language, "engine": "faster-whisper"}

        # Fallback: whisper CLI
        result = subprocess.run(
            ["whisper", tmp_path, "--model", "large-v3", "--output_format", "txt",
             "--output_dir", tempfile.gettempdir(), "--fp16", "False"],
            capture_output=True, text=True, timeout=300
        )
        txt_path = tmp_path.replace(suffix, ".txt")
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                text = clean_transcript(f.read().strip())
            os.unlink(txt_path)
            return {"text": text, "engine": "whisper-cli"}

        return {"error": "Whisper not found. See README.", "stderr": result.stderr}

    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def ollama_request(payload: dict, timeout: int = 120) -> dict:
    # Convert /api/generate-style payload to /api/chat format.
    # Ollama supports both, but /api/chat is more reliable across versions.
    prompt = payload.get("prompt", "")
    chat_payload = {
        "model":   payload.get("model", OLLAMA_MODEL),
        "messages": [{"role": "user", "content": prompt}],
        "stream":  payload.get("stream", True),
        "options": payload.get("options", {}),
    }
    data = json.dumps(chat_payload).encode()

    # Try /api/chat first, fall back to /api/generate if 404
    for endpoint in ("/api/chat", "/api/generate"):
        req = urllib.request.Request(
            f"{OLLAMA_URL}{endpoint}",
            data=data if endpoint == "/api/chat" else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                full_text = ""
                for line in resp:
                    if not line.strip():
                        continue
                    chunk = json.loads(line.decode())
                    # /api/chat puts text in message.content; /api/generate uses response
                    if "message" in chunk:
                        full_text += chunk["message"].get("content", "")
                    else:
                        full_text += chunk.get("response", "")
                    if chunk.get("done"):
                        break
                return {"response": full_text}
        except urllib.error.HTTPError as e:
            if e.code == 404 and endpoint == "/api/chat":
                continue   # try /api/generate
            return {"error": f"Ollama not reachable at {OLLAMA_URL}: HTTP Error {e.code}: {e.reason}"}
        except urllib.error.URLError as e:
            return {"error": f"Ollama not reachable at {OLLAMA_URL}: {e}"}
        except Exception as e:
            return {"error": str(e)}
    return {"error": "Ollama: no working endpoint found"}


def warm_ollama():
    try:
        ollama_request({
            "model": OLLAMA_MODEL,
            "prompt": "hi",
            "stream": True,
            "options": {"num_predict": 1},
        }, timeout=30)
        print("  ✓ Ollama model warmed")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Title + date extraction
# ---------------------------------------------------------------------------

def extract_title_and_date(text: str, fallback_date: str) -> dict:
    prompt = (
        "You are a journal assistant. Read this voice note and return ONLY a JSON object.\n\n"
        "Fields:\n"
        '- "title": 4-8 word evocative diary title capturing mood/theme\n'
        '- "date": day/date mentioned by speaker as "Weekday, Month Day", or null if none\n\n'
        f"Transcript:\n{text[:800]}\n\n"
        'Respond with ONLY valid JSON. Example: {"title": "Couldn\'t sleep, thoughts racing", "date": "Tuesday, May 20"}'
    )
    result = ollama_request({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.2, "num_predict": 60},
    })
    if "error" in result:
        return {"title": None, "date": None, "error": result["error"]}

    raw = result.get("response", "").strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw)
        return {"title": parsed.get("title") or None, "date": parsed.get("date") or None}
    except Exception:
        return {"title": None, "date": None}


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "index.html")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}")

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            with open(FRONTEND_PATH, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif path == "/health":
            whisper_ok     = _whisper_model is not None
            whisper_engine = "faster-whisper" if whisper_ok else "none"
            if not whisper_ok:
                try:
                    r = subprocess.run(["whisper", "--help"], capture_output=True, timeout=5)
                    whisper_ok     = r.returncode == 0
                    whisper_engine = "whisper-cli"
                except Exception:
                    pass
            try:
                urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
                ollama_ok = True
            except Exception:
                ollama_ok = False

            self.send_json({
                "ollama": ollama_ok,
                "whisper": whisper_ok,
                "whisper_engine": whisper_engine,
                "model": OLLAMA_MODEL,
            })
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        path   = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)

        # ── /transcribe ──────────────────────────────────────
        if path == "/transcribe":
            content_type = self.headers.get("Content-Type", "")
            filename     = "audio.m4a"
            audio_bytes  = b""

            if "multipart/form-data" in content_type:
                boundary = content_type.split("boundary=")[-1].encode()
                for part in body.split(b"--" + boundary):
                    if b"Content-Disposition" in part and b"filename=" in part:
                        header_end = part.find(b"\r\n\r\n")
                        if header_end != -1:
                            for line in part[:header_end].decode(errors="replace").splitlines():
                                if "filename=" in line:
                                    fname = line.split("filename=")[-1].strip().strip('"')
                                    if fname:
                                        filename = fname
                            audio_bytes = part[header_end + 4:].rstrip(b"\r\n--")
                            break
            else:
                audio_bytes = body

            if not audio_bytes:
                self.send_json({"error": "No audio data received"}, 400)
                return

            self.send_json(transcribe_audio(audio_bytes, filename))

        # ── /titlize ─────────────────────────────────────────
        elif path == "/titlize":
            try:
                payload = json.loads(body)
                self.send_json(extract_title_and_date(
                    payload.get("text", ""),
                    payload.get("fallback_date", ""),
                ))
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        # ── /summarize ───────────────────────────────────────
        elif path == "/summarize":
            try:
                payload = json.loads(body)
                entries = payload.get("entries", [])
                mode    = payload.get("mode", "week")

                if mode == "entry":
                    prompt = (
                        "You are a personal journal assistant.\n"
                        "Below is a transcribed voice note. Provide:\n"
                        "1. A cleaned-up version (remove filler words, fix grammar)\n"
                        "2. Key themes or action items (if any)\n\n"
                        f"Voice note:\n{entries[0] if entries else ''}"
                    )
                    num_predict = 400
                else:
                    joined = "\n\n".join(
                        f"[{e.get('date','?')}] {e.get('text','')}" for e in entries
                    )
                    prompt = (
                        "You are a personal journal assistant.\n"
                        "These are voice journal entries from this week. Provide:\n"
                        "1. A brief narrative summary (2-3 sentences)\n"
                        "2. Recurring themes or patterns\n"
                        "3. Action items or intentions mentioned\n"
                        "4. One encouraging reflection\n\n"
                        f"Entries:\n{joined}"
                    )
                    num_predict = 600

                self.send_json(ollama_request({
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {"num_predict": num_predict},
                }))
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
        else:
            self.send_json({"error": "Not found"}, 404)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n🎙  Night Notes")
    print(f"   http://localhost:{PORT}")
    print(f"   Ollama model : {OLLAMA_MODEL}\n")

    threading.Thread(target=get_whisper_model, daemon=True).start()
    threading.Thread(target=warm_ollama,       daemon=True).start()

    httpd = ThreadingHTTPServer(("localhost", PORT), Handler)
    print(f"   Press Ctrl+C to stop\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")