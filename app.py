"""
Meeting Copilot - Unified Backend & Server (FastAPI + WebSockets)
Runs the complete loopback capture, faster-whisper STT, RapidFuzz matcher,
Ollama Gemma name alerts, and rolling 60-second meeting summarizer.
"""

import os
import sys
import time
import asyncio
import datetime
import threading
from typing import Set, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure NVIDIA DLL paths are added before imports
from audio_capture import WASAPILoopbackCapture, compute_rms, setup_cuda_dll_paths
setup_cuda_dll_paths()

from faster_whisper import WhisperModel
from config import (
    CHUNK_DURATION_SEC,
    SILENCE_RMS_THRESHOLD,
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE,
    WHISPER_LANGUAGE,
    SUMMARY_INTERVAL_SEC,
    CONTEXT_WINDOW_SEC,
    DEFAULT_USER_NAME,
    HOST,
    PORT,
)
from name_matcher import NameMatcher
from transcript_buffer import TranscriptBuffer
from ollama_worker import OllamaWorker

app = FastAPI(title="Meeting Copilot")

# Core state singletons
transcript_buffer = TranscriptBuffer(max_history_seconds=1800)
name_matcher = NameMatcher()
ollama_worker = OllamaWorker()
latest_summary: Dict[str, Any] = {
    "key_points": [],
    "decisions": [],
    "action_items": [],
    "timestamp": None,
}

# Active WebSocket connections
active_websockets: Set[WebSocket] = set()
ws_lock = asyncio.Lock()
main_loop: Optional[asyncio.AbstractEventLoop] = None

# Audio worker thread control
audio_capturer: Optional[WASAPILoopbackCapture] = None
whisper_model: Optional[WhisperModel] = None
worker_running: bool = False
worker_thread: Optional[threading.Thread] = None
summary_task: Optional[asyncio.Task] = None
last_summary_segment_id: int = 0


async def broadcast_event(event_dict: Dict[str, Any]):
    """Broadcast JSON message to all connected sidebar WebSockets."""
    async with ws_lock:
        to_remove = set()
        for ws in active_websockets:
            try:
                await ws.send_json(event_dict)
            except Exception:
                to_remove.add(ws)
        active_websockets.difference_update(to_remove)


def safe_broadcast_from_thread(event_dict: Dict[str, Any]):
    """Thread-safe bridge to broadcast events from the background audio thread."""
    if main_loop and main_loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast_event(event_dict), main_loop)


def handle_mention_verification(sentence: str, context: str, matched_alias: str, time_str: str):
    """Run Gemma verification in background thread on direct name match."""
    print(f"\n[Direct Match] '{matched_alias}' in '{sentence}'. Asking Gemma 3...")
    result = ollama_worker.verify_mention(sentence, context, DEFAULT_USER_NAME)
    is_directed = result.get("directed", False)
    kind = result.get("kind", "direct_mention")
    reason = result.get("reason", "Direct address to you")

    print(f"[Gemma 3 Verdict] Directed: {is_directed} | Kind: {kind} | Reason: {reason}")

    if is_directed:
        alert_event = {
            "type": "alert",
            "data": {
                "time": time_str,
                "kind": "direct_mention",
                "quote": sentence,
                "reason": reason,
                "alias": matched_alias,
            }
        }
        safe_broadcast_from_thread(alert_event)


def audio_stt_worker():
    """Background worker continuously capturing and transcribing loopback audio."""
    global worker_running, audio_capturer, whisper_model

    print("[Audio STT] Starting background audio capture thread...")
    try:
        audio_capturer = WASAPILoopbackCapture()
        audio_capturer.start_streaming()
    except Exception as e:
        print(f"[ERROR] Failed to start WASAPI loopback capture: {e}")
        return

    # Load faster-whisper model
    try:
        print(f"[STT] Loading faster-whisper '{WHISPER_MODEL}' on {WHISPER_DEVICE} ({WHISPER_COMPUTE})...")
        whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        # Warmup
        _dummy = np.zeros(1600, dtype=np.float32)
        list(whisper_model.transcribe(_dummy, language=WHISPER_LANGUAGE)[0])
        print("[STT] faster-whisper is ready!")
    except Exception as e:
        print(f"[WARNING] Failed to load on CUDA: {e}. Falling back to CPU...")
        whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    target_samples = int(16000 * CHUNK_DURATION_SEC)
    accumulated = []
    current_samples = 0

    while worker_running:
        try:
            chunk = audio_capturer.audio_queue.get(timeout=0.2)
            accumulated.append(chunk)
            current_samples += len(chunk)
        except Exception:
            continue

        if current_samples >= target_samples:
            full_audio = np.concatenate(accumulated)
            accumulated = []
            current_samples = 0

            # Silence gate (save compute when nobody is speaking)
            rms = compute_rms(full_audio)
            if rms < SILENCE_RMS_THRESHOLD:
                continue

            t0 = time.perf_counter()
            try:
                segments, info = whisper_model.transcribe(
                    full_audio,
                    language=WHISPER_LANGUAGE,
                    beam_size=3,
                    vad_filter=True,
                    condition_on_previous_text=False
                )
                text = " ".join(s.text.strip() for s in segments if s.text.strip()).strip()
            except Exception as e:
                print(f"[STT] Transcribe error: {e}")
                continue

            infer_time = time.perf_counter() - t0

            if text:
                # Add to rolling transcript buffer
                entry = transcript_buffer.add_segment(text, infer_time=infer_time)
                print(f"[{entry['timestamp']}] ({infer_time:.2f}s): {text}")

                # Check direct name mentions and group questions
                alert_check = name_matcher.check(text)
                if alert_check["is_alert"]:
                    if alert_check["type"] == "direct":
                        matched_alias = alert_check["alias"]
                        entry["highlight_alias"] = matched_alias
                        context = transcript_buffer.get_context_text(seconds=CONTEXT_WINDOW_SEC)
                        threading.Thread(
                            target=handle_mention_verification,
                            args=(text, context, matched_alias, entry["timestamp"]),
                            daemon=True
                        ).start()
                    elif alert_check["type"] == "group":
                        print(f"\n[Group Alert] Question to everyone: '{text}' (phrase: '{alert_check['matched_phrase']}')")
                        alert_event = {
                            "type": "alert",
                            "data": {
                                "time": entry["timestamp"],
                                "kind": "group_question",
                                "quote": text,
                                "reason": "Open question asked to the room / everyone",
                                "alias": None,
                            }
                        }
                        safe_broadcast_from_thread(alert_event)

                # Broadcast transcript to UI
                transcript_event = {
                    "type": "transcript",
                    "data": entry
                }
                safe_broadcast_from_thread(transcript_event)

    print("[Audio STT] Audio worker loop exited.")


async def rolling_summary_loop():
    """Periodic task: Generates structured summary with Gemma every 60 seconds."""
    global last_summary_segment_id, latest_summary
    print(f"[Summary Engine] Rolling summary loop active (every {SUMMARY_INTERVAL_SEC}s)...")

    while True:
        await asyncio.sleep(SUMMARY_INTERVAL_SEC)

        # Check if there are new transcript chunks
        recent_segments = transcript_buffer.get_recent_segments(seconds=300)
        if not recent_segments:
            continue

        current_max_id = recent_segments[-1]["id"]
        if current_max_id == last_summary_segment_id:
            # No new speech since last summary
            continue

        full_recent_text = transcript_buffer.get_full_text(seconds=300)
        if not full_recent_text.strip():
            continue

        print("\n[Summary Engine] Generating rolling summary with Gemma 3...")
        loop = asyncio.get_running_loop()
        summary_result = await loop.run_in_executor(
            None, ollama_worker.generate_summary, full_recent_text
        )

        if summary_result and (summary_result.get("key_points") or summary_result.get("action_items")):
            latest_summary = summary_result
            last_summary_segment_id = current_max_id
            print(f"[Summary Engine] Summary updated ({len(latest_summary.get('key_points', []))} points, {len(latest_summary.get('action_items', []))} action items)")

            await broadcast_event({
                "type": "summary",
                "data": latest_summary
            })


@app.on_event("startup")
async def startup_event():
    global main_loop, worker_running, worker_thread, summary_task
    main_loop = asyncio.get_running_loop()
    worker_running = True
    worker_thread = threading.Thread(target=audio_stt_worker, daemon=True)
    worker_thread.start()
    summary_task = asyncio.create_task(rolling_summary_loop())


@app.on_event("shutdown")
async def shutdown_event():
    global worker_running, summary_task
    worker_running = False
    if summary_task:
        summary_task.cancel()
    if audio_capturer:
        audio_capturer.stop_streaming()
        audio_capturer.close()


@app.get("/", response_class=HTMLResponse)
async def serve_sidebar():
    """Serve the single-page sidebar UI."""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h3>Meeting Copilot: static/index.html not found</h3>")


@app.get("/api/transcript")
async def get_transcript_export():
    """Export current session transcript in Markdown."""
    return {"markdown": transcript_buffer.export_markdown()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global latest_summary
    await websocket.accept()
    async with ws_lock:
        active_websockets.add(websocket)

    # Send initial state (existing transcript + current summary)
    init_data = {
        "type": "init",
        "data": {
            "transcript": transcript_buffer.get_recent_segments(seconds=300),
            "summary": latest_summary
        }
    }
    await websocket.send_json(init_data)

    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("action") == "summarize":
                # Manual summary trigger from UI button
                full_text = transcript_buffer.get_full_text(seconds=300)
                if full_text.strip():
                    loop = asyncio.get_running_loop()
                    summary_result = await loop.run_in_executor(
                        None, ollama_worker.generate_summary, full_text
                    )
                    latest_summary = summary_result
                    await broadcast_event({
                        "type": "summary",
                        "data": latest_summary
                    })
    except WebSocketDisconnect:
        async with ws_lock:
            active_websockets.discard(websocket)
    except Exception:
        async with ws_lock:
            active_websockets.discard(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False)

