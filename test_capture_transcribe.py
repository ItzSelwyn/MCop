"""
Step 1: Feasibility Benchmark (Capture + Transcribe).
Records N seconds of loopback audio from your speakers/Chrome,
saves it to test_capture.wav, runs faster-whisper, and measures speed.
"""

import os
import sys
import time
import argparse
import numpy as np

from audio_capture import WASAPILoopbackCapture, compute_rms, setup_cuda_dll_paths
setup_cuda_dll_paths()

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


def parse_args():
    parser = argparse.ArgumentParser(description="Test WASAPI loopback capture and faster-whisper transcription.")
    parser.add_argument("--duration", type=int, default=15, help="Duration to record in seconds (default: 15s). Try 60s for full test.")
    parser.add_argument("--model", type=str, default="small", help="faster-whisper model size (tiny, base, small, medium). Default: small.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device (default: auto).")
    parser.add_argument("--save-wav", type=str, default="test_capture.wav", help="WAV file to save recorded audio.")
    return parser.parse_args()


def get_whisper_model(model_size: str, requested_device: str):
    """
    Load WhisperModel with graceful fallback from CUDA to CPU.
    """
    if WhisperModel is None:
        print("\n[ERROR] faster-whisper is not installed.")
        print("Run: pip install faster-whisper\n")
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
            print(f"Loading '{model_size}' model on {dev.upper()} ({compute})... (downloading weights if first run)")
            t0 = time.perf_counter()
            model = WhisperModel(model_size, device=dev, compute_type=compute)
            # Warmup inference to verify CUDA execution and DLL loading immediately
            _dummy = np.zeros(1600, dtype=np.float32)
            _segs, _ = model.transcribe(_dummy, language="en")
            list(_segs)
            load_time = time.perf_counter() - t0
            print(f"[OK] Loaded successfully in {load_time:.2f}s on {dev.upper()} ({compute}).")
            return model, dev, compute, load_time
        except Exception as e:
            print(f"[NOTE] Could not load on {dev.upper()} ({compute}): {e}")
            if dev == "cuda" and requested_device == "auto":
                print("Falling back to next available option...")

    raise RuntimeError("Failed to load WhisperModel on any device.")


def main():
    args = parse_args()

    print("=" * 65)
    print("      MEETING COPILOT - CAPTURE + TRANSCRIBE BENCHMARK")
    print("=" * 65)
    print(f"Test Configuration:")
    print(f"  Duration : {args.duration} seconds")
    print(f"  Model    : {args.model}")
    print(f"  Device   : {args.device}")
    print("-" * 65)

    # 1. Initialize Capture
    print("Initializing WASAPI Loopback Capture...")
    try:
        capturer = WASAPILoopbackCapture()
    except Exception as e:
        print(f"[ERROR] Failed to initialize loopback audio: {e}")
        sys.exit(1)

    device_name = capturer.device_info["name"]
    native_rate = capturer.native_rate
    channels = capturer.channels
    print(f"[OK] Capturing from: '{device_name}'")
    print(f"     Native Format : {native_rate} Hz, {channels} channel(s)")

    print("\n" + ">" * 10 + " INSTRUCTIONS " + "<" * 10)
    print("Play some audio RIGHT NOW (a YouTube video, meeting clip, podcast, etc.)")
    print("so there is actual speech to capture!")
    print("Recording will start in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"  Starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    print("  Recording NOW!                  ", flush=True)
    print("-" * 65)

    # 2. Record Loopback Audio
    t_start = time.perf_counter()
    audio_16k = capturer.record_chunk(args.duration)
    record_time = time.perf_counter() - t_start
    capturer.close()

    rms_val = compute_rms(audio_16k)
    print(f"[OK] Captured {len(audio_16k) / 16000:.1f}s of audio in {record_time:.2f}s (Audio RMS: {rms_val:.5f})")

    if rms_val < 0.001:
        print("\n[WARNING] The recorded audio is nearly silent (RMS < 0.001)!")
        print("          Was sound playing through your speakers during recording?")

    # 3. Save WAV file for user verification
    if sf is not None and args.save_wav:
        sf.write(args.save_wav, audio_16k, 16000)
        print(f"[OK] Saved raw audio to '{args.save_wav}'. You can listen to it to verify sound was captured.")

    # 4. Load Whisper Model
    print("-" * 65)
    model, device_used, compute_used, load_time = get_whisper_model(args.model, args.device)

    # 5. Transcribe
    print(f"Transcribing {args.duration}s of audio...")
    t_transcribe_start = time.perf_counter()
    segments, info = model.transcribe(
        audio_16k,
        language="en",
        beam_size=5,
        vad_filter=True,
    )
    segments = list(segments)
    transcribe_time = time.perf_counter() - t_transcribe_start

    # 6. Performance Metrics
    actual_audio_dur = len(audio_16k) / 16000.0
    speed_factor = actual_audio_dur / max(transcribe_time, 0.0001)
    rtf = transcribe_time / actual_audio_dur

    print("-" * 65)
    print("BENCHMARK RESULTS:")
    print(f"  Audio Duration      : {actual_audio_dur:.2f}s")
    print(f"  Transcription Time  : {transcribe_time:.2f}s")
    print(f"  Speed Multiplier    : {speed_factor:.1f}x real-time (higher is better)")
    print(f"  Real-Time Factor    : {rtf:.3f} (lower is better, < 1.0 means faster than real-time)")
    print(f"  Device & Compute    : {device_used.upper()} ({compute_used})")
    print(f"  Detected Language   : {info.language} (prob: {info.language_probability:.2f})")
    print("-" * 65)

    print("TRANSCRIPTION OUTPUT:")
    if len(segments) == 0:
        print("  [No speech detected or audio was silent]")
    else:
        for s in segments:
            print(f"  [{s.start:05.2f}s -> {s.end:05.2f}s] {s.text.strip()}")

    print("=" * 65)
    if speed_factor >= 2.0:
        print("[FEASIBILITY VERDICT: PASS]")
        print(f"Your PC transcribes at {speed_factor:.1f}x real-time speed!")
        print("You have more than enough performance headroom for live meeting streaming.")
    elif speed_factor >= 1.0:
        print("[FEASIBILITY VERDICT: ACCEPTABLE]")
        print(f"Transcription runs at {speed_factor:.1f}x real-time.")
        print("For smoother live performance, consider --model base or enabling GPU acceleration.")
    else:
        print("[FEASIBILITY VERDICT: SLOW]")
        print(f"Transcription is slower than real-time ({speed_factor:.2f}x).")
        print("Recommendation: Use --model tiny or --model base, or ensure CUDA GPU acceleration is enabled.")
    print("=" * 65)


if __name__ == "__main__":
    main()

