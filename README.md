# 🎙 Night Notes

**A fully local, fully private voice journal.** Drop an audio file → get a transcript, an auto-generated title, and a weekly reflection — all running on your own machine. Nothing leaves your laptop.

---

## What it does

- **Transcribes** voice notes using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (Whisper large-v3, runs offline)
- **Handles code-switching** — works if you mix languages mid-sentence (Hindi/English, Spanish/English, etc.)
- **Auto-titles** each note using a local Ollama model
- **Infers dates** from what you say ("so today was a rough Tuesday…")
- **Weekly summaries** — one click to get a narrative reflection of your week
- **Fully local** — no API keys, no cloud, no accounts

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/download) with a Qwen model (or any model you prefer)
- ~1.5 GB disk space for the Whisper large-v3 model

---

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/night-notes.git
cd night-notes
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Check Python and pip
2. Install `faster-whisper`
3. Detect your Ollama installation and available Qwen model
4. Auto-patch `server.py` with your model name
5. Pre-download the Whisper large-v3 model (~1.5 GB, one-time)

Then start the server:

```bash
python3 server.py
```

Open **http://localhost:8765** in your browser. The status pills turn green once Whisper and Ollama are ready (they load in the background — usually under 10 seconds).

---

## iPhone → Mac transfer (pick one)

| Method | Effort | Speed |
|--------|--------|-------|
| **iCloud + Voice Memos** ⭐ | Zero (auto-syncs) | ~30s |
| **AirDrop** | One tap after recording | Instant |
| **Just Press Record app** | One-time setup, then automatic | Automatic |
| **USB/Finder** | Plug in, drag | Fast |

**Recommended:** Record in Apple's Voice Memos → it syncs automatically to iCloud Drive → open Finder on Mac → drag the week's files into Night Notes in one batch.

---

## Performance

The server is optimised for fast turnaround on CPU:

| What | How |
|------|-----|
| Whisper model loaded once at startup | No reload per file |
| `beam_size=2` instead of 5 | ~2× faster, near-identical quality for speech |
| Tight VAD silence padding | Fewer wasted frames |
| `ThreadingHTTPServer` | Transcription + titlize run concurrently |
| Ollama warmed at startup | First request hits a hot model |

---

## Configuration

Edit the top of `server.py`:

```python
PORT         = 8765          # port the server runs on
OLLAMA_MODEL = "qwen2.5:14b"     # auto-set by setup.sh; change to any `ollama list` model
```

To use a smaller/faster Whisper model (less accurate but quicker):

```python
_whisper_model = WhisperModel("medium", ...)   # or "small", "base"
```

---

## Project structure

```
night-notes/
├── server.py        # Python backend — Whisper + Ollama proxy
├── index.html       # Browser UI (served by server.py at localhost:8765)
├── setup.sh         # One-command dependency installer
├── requirements.txt # Python dependencies
└── README.md
```

---

## How the stack works

```
Browser (index.html)
    │
    ├── POST /transcribe  →  faster-whisper large-v3 (local)
    ├── POST /titlize     →  Ollama  (extract title + date from transcript)
    └── POST /summarize   →  Ollama  (weekly narrative / entry cleanup)
```

All entries are stored in your browser's `localStorage`. To back up your journal, open DevTools → Application → Local Storage → copy the `nj_entries` value.

---

## License

MIT
