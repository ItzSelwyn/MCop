"""
Step 2: Live Terminal Transcript Loop.
Continuously captures desktop/Chrome audio via WASAPI loopback,
processes rolling audio chunks with faster-whisper,
and prints a live transcript to the terminal in real time.
"""

import os
import sys
import time
import argparse
import datetime
import numpy as np

from audio_capture import WASAPILoopbackCapture, compute_rms, setup_cuda_dll_paths
setup_cuda_dll_paths()

try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


def color_text(text: str, color_code: str) -> str:
    if HAS_COLOR:
        return f"{color_code}{text}{Style.RESET_ALL}"
    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Live WASAPI loopback transcription in your terminal.")
    parser.add_argument("--model", type=str, default="small", choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size (default: small).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Compute device (default: auto).")
    parser.add_argument("--chunk", type=float, default=5.0,
                        help="Audio chunk duration in seconds before transcribing (default: 5.0s).")
    parser.add_argument("--silence-threshold", type=float, default=0.003,
                        help="RMS threshold below which audio is treated as silence (default: 0.003).")
    parser.add_argument("--output", type=str, default="transcript.txt",
                        help="File path to continuously save transcript (default: transcript.txt).")
    return parser.parse_args()


def load_model(model_size: str, requested_device: str):
    """Load WhisperModel with CUDA -> CPU fallback."""
    if WhisperModel is None:
        print("\n[ERROR] faster-whisper is not installed. Run: pip install faster-whisper\n")
        sys.exit(1)

    devices_to_try = []
    if requested_device == "auto":
        devices_to_try = [("cuda", "float16"), ("cuda", "int8_float16"), ("cpu", "int8")]
    elif requested_device == "cuda":
        devices_to_try = [("cuda", "float16"), ("cuda", "int8_float16")]
    else:
        devices_to_try = [("cpu", "int8")]

    for dev, compute in devices_to_try:
        try:
            print(f"Loading '{model_size}' model on {dev.upper()} ({compute})...")
            t0 = time.perf_counter()
            model = WhisperModel(model_size, device=dev, compute_type=compute)
            # Warmup to verify DLLs and CUDA runtime
            _dummy = np.zeros(1600, dtype=np.float32)
            _segs, _ = model.transcribe(_dummy, language="en")
            list(_segs)
            load_time = time.perf_counter() - t0
            print(f"[OK] Loaded in {load_time:.2f}s on {dev.upper()} ({compute}).")
            return model, dev, compute
        except Exception as e:
            print(f"[NOTE] Could not load on {dev.upper()} ({compute}): {e}")
            if dev == "cuda" and requested_device == "auto":
                print("Falling back to next available option...")

    raise RuntimeError("Failed to load WhisperModel.")


def main():
    args = parse_args()

    print("=" * 65)
    print("      MEETING COPILOT - LIVE TERMINAL TRANSCRIPT")
    print("=" * 65)
    print(f"  Model             : {args.model}")
    print(f"  Chunk Size        : {args.chunk}s")
    print(f"  Silence Threshold : {args.silence_threshold}")
    print(f"  Transcript Output : {args.output}")
    print("-" * 65)

    # 1. Load Model
    model, device_used, compute_used = load_model(args.model, args.device)

    # 2. Setup WASAPI Loopback Capture
    print("Initializing WASAPI Loopback Capture...")
    try:
        capturer = WASAPILoopbackCapture()
    except Exception as e:
        print(f"[ERROR] Failed to initialize loopback audio: {e}")
        sys.exit(1)

    print(f"[OK] Capturing audio from: '{capturer.device_info['name']}'")
    print("-" * 65)
    print(color_text("Listening for meeting audio... Press Ctrl+C to stop.", Fore.CYAN if HAS_COLOR else ""))
    print("-" * 65)

    capturer.start_streaming()

    # Audio chunk accumulation buffer
    target_samples_per_chunk = int(16000 * args.chunk)
    audio_buffer = []
    current_sample_count = 0
    total_transcribed_lines = 0

    output_file = open(args.output, "a", encoding="utf-8")
    output_file.write(f"\n--- Meeting Session Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    output_file.flush()

    try:
        while True:
            try:
                # Read 16kHz chunk from capture queue (with timeout so Ctrl+C stays responsive)
                chunk = capturer.audio_queue.get(timeout=0.2)
                audio_buffer.append(chunk)
                current_sample_count += len(chunk)
            except Exception:
                continue

            if current_sample_count >= target_samples_per_chunk:
                # Combine accumulated chunks
                full_audio = np.concatenate(audio_buffer)
                audio_buffer = []
                current_sample_count = 0

                # Check energy / silence
                rms = compute_rms(full_audio)
                if rms < args.silence_threshold:
                    # Print subtle progress heartbeat so user knows it's actively listening
                    sys.stdout.write(".")
                    sys.stdout.flush()
                    continue

                # Run transcription
                t0 = time.perf_counter()
                segments, info = model.transcribe(
                    full_audio,
                    language="en",
                    beam_size=3,
                    vad_filter=True,
                    condition_on_previous_text=False
                )
                segment_list = list(segments)
                infer_time = time.perf_counter() - t0

                # Filter and print detected speech
                speech_text = " ".join(s.text.strip() for s in segment_list if s.text.strip()).strip()
                if speech_text:
                    now_str = datetime.datetime.now().strftime("%H:%M:%S")
                    time_tag = color_text(f"[{now_str}]", Fore.GREEN if HAS_COLOR else "")
                    latency_tag = color_text(f"({infer_time:.2f}s)", Fore.YELLOW if HAS_COLOR else "")
                    clean_line = f"{time_tag} {latency_tag} {speech_text}"
                    print(f"\n{clean_line}")

                    # Save to file
                    file_line = f"[{now_str}] {speech_text}\n"
                    output_file.write(file_line)
                    output_file.flush()
                    total_transcribed_lines += 1

    except KeyboardInterrupt:
        print("\n\n" + "-" * 65)
        print("Stopping live capture...")
    finally:
        capturer.stop_streaming()
        capturer.close()
        output_file.write(f"--- Session Ended: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")
        output_file.close()
        print(f"[DONE] Capture stopped. Logged {total_transcribed_lines} transcript lines to '{args.output}'.")
        print("=" * 65)


if __name__ == "__main__":
    main()

