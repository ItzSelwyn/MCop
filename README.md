# Meeting Copilot

> **A privacy-first, ultra-lightweight meeting assistant that runs 100% locally on your laptop.**  
> Captures system audio from Google Meet, Microsoft Teams, or Zoom web meetings, transcribes speech in real time with GPU acceleration, detects when questions are directed at you or the team, and generates rolling structured meeting summaries — all displayed in a native minimalist Chrome sidebar window.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Whisper](https://img.shields.io/badge/STT-faster--whisper-orange?style=flat)](https://github.com/SYSTRAN/faster-whisper)
[![Ollama](https://img.shields.io/badge/LLM-Gemma%203%201B%20(Ollama)-black?style=flat&logo=ollama&logoColor=white)](https://ollama.com/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20WASAPI-blue?style=flat&logo=windows&logoColor=white)](https://github.com/intxcc/pyaudiowpatch)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📺 Live Demo Video

Watch Meeting Copilot in action: capturing live audio, transcribing speech with sub-second latency, detecting questions, and generating rolling summaries:

👉 **[Watch the Demo Video (Google Drive)](https://drive.google.com/file/d/1Z7-UFpXl-HhTpTzdhHhKGaLFAMVMDi8x/view?usp=sharing)**

---

## ⚡ Highlights

- **Zero Virtual Cables & No Extensions**: Captures directly from Windows WASAPI loopback (`PyAudioWPatch`). Works with any meeting platform in Chrome, desktop apps, or media player.
- **GPU-Accelerated Transcription**: Runs `faster-whisper` on your local GPU (e.g. RTX 3050 Laptop) in `float16` / `int8_float16`, transcribing 15 seconds of speech in ~1.1 seconds (**12x – 25x faster than real-time**).
- **Dual-Mode Attention Alerts**:
  - **`DIRECT TO YOU`**: Uses `RapidFuzz` for microsecond phonetic name matching, followed by a grounded Gemma 3 evaluation on the last 35 seconds of context to verify that you are being asked a question or assigned a task.
  - **`QUESTION TO ALL`**: Real-time heuristic detection for open room questions (*"did you all..."*, *"can anyone..."*, *"does anyone have questions..."*, *"what does everyone think?"*) so you never sit on mute when the team is waiting for an answer.
- **Synthesized Audio Chime**: An alert sound is generated natively in the browser via the Web Audio API (zero audio asset dependencies, works 100% offline).
- **Rolling 60-Second Summaries**: Gemma 3 automatically aggregates recent conversation into structured JSON containing **Key Points**, **Decisions**, and **Action Items** (with checkboxes and assignee tags).
- **Minimalist Professional Dark UI**: Clean monochrome palette (`#09090b` / neutral dark zinc), vector SVG icons, flat surfaces, zero gradients, and zero colorful emojis.
- **Single Process + Ollama**: Everything runs on your machine across one Python process and your local Ollama daemon. No cloud API subscriptions or external telemetry.

---

## 🏗️ Architecture

```
Chrome / Teams / Zoom Web Meeting
        │
        ▼ (Audio output to speakers/headphones)
[1] WASAPI Loopback Capture (PyAudioWPatch)
        │  16 kHz mono downmixed audio chunks
        ▼
[2] faster-whisper "small" (NVIDIA RTX GPU)
        │  Sub-second transcription per ~5s chunk
        ▼
[3] Rolling Transcript Buffer (Thread-safe memory store)
        │
        ├──► [4a] Name & Alias Matcher (RapidFuzz)
        │           └─ Hit ─► [5a] Gemma 3 1B via Ollama (CPU)
        │                            └─► ALERT: DIRECT TO YOU (+ 35s context)
        │
        ├──► [4b] Group Question Detector (Regex Heuristics)
        │           └─ Hit ──────────────────────────────────────────► ALERT: QUESTION TO ALL
        │
        └──► [6] Every 60s: Gemma 3 1B Summary Worker
                    └─ JSON: Key Points / Decisions / Action Items
        ▼
[7] FastAPI + WebSocket Server (app.py)
        │  Live event broadcast
        ▼
[8] Native Sidebar Window (Chrome --app=http://127.0.0.1:8000)
```

---

## 📊 Hardware Benchmarks

Tested on an **NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)** with **Python 3.11**:

| Component | Target Device | Memory Usage | Latency / Speed |
| :--- | :--- | :--- | :--- |
| **Audio Loopback** | Windows WASAPI | < 15 MB RAM | ~0 ms capture latency |
| **faster-whisper (`small`)** | NVIDIA RTX 3050 (float16) | ~850 MB VRAM | **12.8x faster than real-time** (15s audio in 1.18s) |
| **Gemma 3 1B** | Local CPU (via Ollama) | ~1.1 GB System RAM | ~0.8s for alert classification, ~2.2s for 60s summary |
| **FastAPI + UI** | Localhost WebSockets | ~45 MB RAM | < 5 ms client broadcast |

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- **Operating System**: Windows 10 or Windows 11
- **Python**: Python 3.10 or 3.11 (Tested with 3.11.0)
- **Ollama**: [Download & Install Ollama](https://ollama.com/download)

### 2. Pull the Gemma 3 Model
Open PowerShell or Command Prompt and run:
```powershell
ollama pull gemma3:1b
```

### 3. Setup Virtual Environment
In the project directory:
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts scripts, execute `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once).*

### 4. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. (Recommended) Enable NVIDIA GPU Acceleration
To run `faster-whisper` on your NVIDIA GPU with CUDA 12 without installing the full CUDA Toolkit:
```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

---

## 💻 Running Meeting Copilot

### One-Click Launch
Start both the backend server and the desktop Chrome sidebar in one command:

```powershell
python launch_copilot.py
```
*(Or simply double-click **`start_copilot.bat`** in Windows Explorer).*

This command will:
1. Initialize WASAPI loopback capture from your default playback device.
2. Load `faster-whisper` on your GPU.
3. Establish connection to local Ollama (`gemma3:1b`).
4. Spawn a slim, standalone Chrome app window (`--app=http://127.0.0.1:8000 --window-size=410,820`) next to your meeting.

---

## 🛠️ Step-by-Step Diagnostic & Benchmark Tools

The codebase provides standalone diagnostic scripts to test individual components:

### Test 0: Audio Device Diagnostic
Verifies WASAPI loopback compatibility and captures a 1-second sample:
```powershell
python check_audio_devices.py
```

### Test 1: Feasibility & Transcription Benchmark
Plays audio in Chrome, records 15 seconds from your speakers, saves `test_capture.wav`, and benchmarks GPU transcription speed:
```powershell
python test_capture_transcribe.py --duration 15
```

### Test 2: Live Terminal-Only Stream
Streams real-time speech transcription directly into your terminal:
```powershell
python live_transcribe.py
```

---

## ⚙️ Configuration (`config.py`)

All settings can be customized in [config.py](file:///d:/coding/MCop/config.py):

```python
# Target user for direct alert filtering
DEFAULT_USER_NAME: str = "Selwyn"

# Phonetic and misrecognition aliases (evaluated with RapidFuzz >= 82 score)
DEFAULT_ALIASES: List[str] = ["selwyn", "selvin", "selwin", "salwin"]

# Short nickname tokens requiring exact word boundary matching
EXACT_ALIASES: List[str] = ["sel"]

# Toggle open room/team question alerts
ALERT_ON_GROUP_QUESTIONS: bool = True

# Audio & Model Parameters
CHUNK_DURATION_SEC: float = 5.0       # Audio buffer chunk length
SILENCE_RMS_THRESHOLD: float = 0.003  # Silence gate to save compute
WHISPER_MODEL: str = "small"          # tiny, base, small, medium
WHISPER_DEVICE: str = "cuda"          # cuda or cpu
WHISPER_COMPUTE: str = "float16"      # float16, int8_float16, or int8
OLLAMA_MODEL: str = "gemma3:1b"       # LLM model tag
SUMMARY_INTERVAL_SEC: int = 60        # Seconds between automatic rolling summaries
```

---

## 📁 Repository Structure

```
MCop/
├── config.py                  # User configuration, aliases, model settings, and thresholds
├── audio_capture.py           # WASAPI loopback audio stream & 16 kHz polyphase resampler
├── name_matcher.py            # RapidFuzz name matcher & group question pattern engine
├── ollama_worker.py           # Gemma 3 client: directed mention filter & structured JSON summaries
├── transcript_buffer.py       # Thread-safe rolling memory buffer with context extraction
├── app.py                     # FastAPI application, background audio thread, WebSocket broadcaster
├── static/
│   └── index.html             # Minimalist dark-mode sidebar UI (HTML5, Vanilla JS, CSS)
├── launch_copilot.py          # One-click launcher (backend + standalone Chrome app window)
├── start_copilot.bat          # Double-click launcher script for Windows
├── check_audio_devices.py     # WASAPI device inspector and audio test script
├── test_capture_transcribe.py # Step 1 feasibility & benchmark test script
├── live_transcribe.py         # Terminal-only live streaming transcriber
└── requirements.txt           # Core Python dependencies
```

---

## 🔒 Privacy & Local Processing Guarantee

- **No Cloud STT**: All audio stays in your computer's RAM and GPU VRAM.
- **No Cloud LLM**: Gemma 3 runs strictly inside your local Ollama instance on CPU.
- **Zero Browser Extensions**: WASAPI captures at the OS loopback layer; no permissions or access to browser history or cookies required.
- **No Audio Saved to Disk**: Audio frames are processed in-memory as rolling 16 kHz numpy arrays and discarded after transcription.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
