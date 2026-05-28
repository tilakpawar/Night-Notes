#!/usr/bin/env python3
"""
Audio Journal Local Server
- Transcribes audio using faster-whisper (local)
- Proxies to Ollama for summarization
- Serves the frontend HTML
"""

import os
import json
import tempfile
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import subprocess
import sys

PORT = 8765
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.7"  # adjust if your model name differs

# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def transcribe_audio(audio_bytes: bytes, filename: str) -> dict:
    """Transcribe audio bytes using faster-whisper (preferred) or whisper."""
    suffix = os.path.splitext(filename)[-1] or ".m4a"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        # Try faster-whisper first
        try:
            from faster_whisper import WhisperModel
            # large-v3 is required for reliable code-switching (e.g. Hindi/English).
            # base/small lock onto one language and silently drop sentences in others.
            # int8 keeps it fast on CPU; use "float16" if you have a GPU.
            model = WhisperModel("large-v3", device="cpu", compute_type="int8")
            segments, info = model.transcribe(
                tmp_path,
                beam_size=5,
                language=None,        # auto-detect per segment, not once for the whole file
                task="transcribe",
                multilingual=True,    # enable code-switching across languages
                vad_filter=True,      # skip silent gaps (speeds things up)
            )
            text = " ".join(seg.text.strip() for seg in segments)
            return {"text": text, "language": info.language, "engine": "faster-whisper"}
        except ImportError:
            pass

        # Fallback: whisper CLI (also passes --language flag as None for multilingual)
        result = subprocess.run(
            ["whisper", tmp_path, "--model", "large-v3", "--output_format", "txt",
             "--output_dir", tempfile.gettempdir(), "--fp16", "False"],
            capture_output=True, text=True, timeout=300
        )
        txt_path = tmp_path.replace(suffix, ".txt")
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                text = f.read().strip()
            os.unlink(txt_path)
            return {"text": text, "engine": "whisper-cli"}

        return {"error": "Whisper not found. See setup instructions.", "stdout": result.stdout, "stderr": result.stderr}

    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Ollama proxy
# ---------------------------------------------------------------------------

def ollama_request(payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            # Ollama streams newline-delimited JSON; collect full response
            full_text = ""
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line.decode())
                full_text += chunk.get("response", "")
                if chunk.get("done"):
                    break
            return {"response": full_text}
    except urllib.error.URLError as e:
        return {"error": f"Ollama not reachable at {OLLAMA_URL}: {e}"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Title + date extraction
# ---------------------------------------------------------------------------

def extract_title_and_date(text: str, fallback_date: str) -> dict:
    """Ask Ollama to pull a title and any mentioned date from the transcript."""
    prompt = f"""You are a journal assistant. Read this voice note transcript and return ONLY a JSON object — no explanation, no markdown, no backticks.

Extract:
1. "title": A short, evocative title (4-8 words) that captures the mood or main theme of this note. Make it feel personal and diary-like, not generic.
2. "date": If the speaker mentions a day or date (e.g. "today is Tuesday", "it's the 14th", "this Monday"), return it as a human-readable string like "Tuesday, May 14". If no date is mentioned, return null.

Transcript:
{text[:1200]}

Respond with ONLY valid JSON, example: {{"title": "Tired but hopeful after the meeting", "date": "Monday, May 12"}}"""

    result = ollama_request({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.3, "num_predict": 80}
    })

    if "error" in result:
        return {"title": None, "date": None, "error": result["error"]}

    raw = result.get("response", "").strip()
    # Strip accidental markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(raw)
        return {
            "title": parsed.get("title") or None,
            "date": parsed.get("date") or None,
        }
    except Exception:
        # Best-effort: try to find something title-like in the raw response
        return {"title": None, "date": None, "raw": raw}


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

FRONTEND_PATH = os.path.join(os.path.dirname(__file__), "index.html")

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}")

    def send_json(self, data: dict, status=200):
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
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            with open(FRONTEND_PATH, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif parsed.path == "/health":
            # Check Ollama
            try:
                urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
                ollama_ok = True
            except Exception:
                ollama_ok = False

            # Check whisper
            whisper_ok = False
            whisper_engine = "none"
            try:
                from faster_whisper import WhisperModel
                whisper_ok = True
                whisper_engine = "faster-whisper"
            except ImportError:
                try:
                    r = subprocess.run(["whisper", "--help"], capture_output=True, timeout=5)
                    whisper_ok = r.returncode == 0
                    whisper_engine = "whisper-cli"
                except Exception:
                    pass

            self.send_json({
                "ollama": ollama_ok,
                "whisper": whisper_ok,
                "whisper_engine": whisper_engine,
                "model": OLLAMA_MODEL
            })
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        if parsed.path == "/transcribe":
            # Expect multipart; extract audio bytes naively
            content_type = self.headers.get("Content-Type", "")
            filename = "audio.m4a"
            audio_bytes = b""

            if "multipart/form-data" in content_type:
                boundary = content_type.split("boundary=")[-1].encode()
                parts = body.split(b"--" + boundary)
                for part in parts:
                    if b"Content-Disposition" in part and b"filename=" in part:
                        # Extract filename
                        header_end = part.find(b"\r\n\r\n")
                        if header_end != -1:
                            headers_raw = part[:header_end].decode(errors="replace")
                            for line in headers_raw.splitlines():
                                if "filename=" in line:
                                    fname_part = line.split("filename=")[-1].strip().strip('"')
                                    if fname_part:
                                        filename = fname_part
                            audio_bytes = part[header_end + 4:].rstrip(b"\r\n--")
                            break
            else:
                audio_bytes = body

            if not audio_bytes:
                self.send_json({"error": "No audio data received"}, 400)
                return

            result = transcribe_audio(audio_bytes, filename)
            self.send_json(result)

        elif parsed.path == "/titlize":
            try:
                payload = json.loads(body)
                text = payload.get("text", "")
                fallback_date = payload.get("fallback_date", "")
                result = extract_title_and_date(text, fallback_date)
                self.send_json(result)
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        elif parsed.path == "/summarize":
            try:
                payload = json.loads(body)
                entries = payload.get("entries", [])
                mode = payload.get("mode", "week")  # "week" or "entry"

                if mode == "entry":
                    prompt = f"""You are a thoughtful personal journal assistant. 
Below is a transcribed voice note. Clean it up (fix filler words, grammar) and provide:
1. A cleaned-up version of the note
2. Key themes or action items (if any)

Keep it concise and personal. Voice note:
{entries[0] if entries else ''}"""
                else:
                    joined = "\n\n".join(
                        f"[{e.get('date','?')}] {e.get('text','')}" for e in entries
                    )
                    prompt = f"""You are a thoughtful personal journal assistant.
Below are voice journal entries from this week. Please provide:
1. A brief narrative summary of the week (2-3 sentences)
2. Recurring themes or patterns you notice
3. Any action items or intentions mentioned
4. One encouraging reflection

Entries:
{joined}"""

                result = ollama_request({"model": OLLAMA_MODEL, "prompt": prompt, "stream": True})
                self.send_json(result)
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
        else:
            self.send_json({"error": "Not found"}, 404)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n🎙  Audio Journal Server")
    print(f"   http://localhost:{PORT}\n")
    print(f"   Ollama model : {OLLAMA_MODEL}")
    print(f"   Press Ctrl+C to stop\n")
    httpd = HTTPServer(("localhost", PORT), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
