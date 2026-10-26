"""
Audio Capture Module using WASAPI Loopback (PyAudioWPatch).
Captures desktop / system sound (Chrome, Teams, Zoom, Spotify, etc.)
and outputs 16 kHz mono float32 numpy arrays ready for faster-whisper.
"""

import os
import sys
import time
import math
import queue
import threading
from typing import Optional, Tuple, Dict, Any

import numpy as np
import scipy.signal

# Windows DLL path resolution for NVIDIA CUDA/cuDNN packages installed via pip
def setup_cuda_dll_paths():
    """Add NVIDIA pip package DLL directories to Python search path and PATH on Windows."""
    # Suppress HuggingFace symlink warning on Windows
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    if sys.platform != "win32":
        return
    for path in sys.path:
        if "site-packages" in path:
            nvidia_dir = os.path.join(path, "nvidia")
            if os.path.isdir(nvidia_dir):
                for root, _, files in os.walk(nvidia_dir):
                    if any(f.lower().endswith(".dll") for f in files):
                        try:
                            os.add_dll_directory(root)
                        except Exception:
                            pass
                        # CTranslate2 / Win32 LoadLibrary specifically requires the folder in PATH
                        if root not in os.environ.get("PATH", ""):
                            os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")

# Ensure CUDA DLLs are found before ctranslate2 is imported
setup_cuda_dll_paths()

try:
    import pyaudiowpatch as pyaudio
except ImportError:
    pyaudio = None


def resample_to_16k(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """
    Resample 1D float32 audio from orig_sr to target_sr using polyphase filtering.
    """
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    
    gcd_val = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd_val
    down = orig_sr // gcd_val
    resampled = scipy.signal.resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def compute_rms(audio: np.ndarray) -> float:
    """Compute Root Mean Square (RMS) energy of audio array."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


class WASAPILoopbackCapture:
    """
    Manages capturing system audio through Windows WASAPI loopback.
    """
    def __init__(self, device_index: Optional[int] = None):
        if pyaudio is None:
            raise ImportError(
                "PyAudioWPatch is not installed. Run: pip install PyAudioWPatch"
            )
        self.p = pyaudio.PyAudio()
        self.device_info = self._resolve_device(device_index)
        self.native_rate = int(self.device_info["defaultSampleRate"])
        self.channels = int(self.device_info["maxInputChannels"])
        self.stream = None
        self._running = False
        self._thread = None
        self.audio_queue = queue.Queue()

    def _resolve_device(self, device_index: Optional[int]) -> Dict[str, Any]:
        """Find the requested or default WASAPI loopback device."""
        try:
            wasapi_info = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            raise RuntimeError("WASAPI is not available on this Windows system.")

        if device_index is not None:
            info = self.p.get_device_info_by_index(device_index)
            if not info.get("isLoopbackDevice", False):
                print(f"[Warning] Device #{device_index} is not marked as a loopback device.")
            return info

        # Auto-detect default loopback device
        try:
            default_loopback = self.p.get_default_wasapi_loopback()
            if default_loopback:
                return default_loopback
        except Exception:
            pass

        # Fallback: scan loopback devices for default output device match
        try:
            default_output = self.p.get_default_output_device_info()
            for dev in self.p.get_loopback_device_info_generator():
                if default_output["name"] in dev["name"]:
                    return dev
        except Exception:
            pass

        # Fallback: pick the first available loopback device
        for dev in self.p.get_loopback_device_info_generator():
            return dev

        raise RuntimeError(
            "No WASAPI loopback audio device found! Make sure speakers or headphones are enabled."
        )

    def record_chunk(self, duration_sec: float) -> np.ndarray:
        """
        Record a fixed duration of audio and return a 1D float32 numpy array at 16,000 Hz.
        """
        frames_per_buffer = 1024
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.native_rate,
            input=True,
            input_device_index=self.device_info["index"],
            frames_per_buffer=frames_per_buffer,
        )

        total_frames = int(self.native_rate * duration_sec)
        frames_read = 0
        raw_chunks = []

        try:
            while frames_read < total_frames:
                to_read = min(frames_per_buffer, total_frames - frames_read)
                data = stream.read(to_read, exception_on_overflow=False)
                raw_chunks.append(data)
                frames_read += to_read
        finally:
            stream.stop_stream()
            stream.close()

        raw_bytes = b"".join(raw_chunks)
        raw_int16 = np.frombuffer(raw_bytes, dtype=np.int16)

        if self.channels > 1:
            raw_int16 = raw_int16.reshape(-1, self.channels)
            mono_float = raw_int16.mean(axis=1).astype(np.float32) / 32768.0
        else:
            mono_float = raw_int16.astype(np.float32) / 32768.0

        return resample_to_16k(mono_float, self.native_rate, 16000)

    def start_streaming(self):
        """Start background loopback capture streaming into self.audio_queue."""
        if self._running:
            return
        self._running = True
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.native_rate,
            input=True,
            input_device_index=self.device_info["index"],
            frames_per_buffer=2048,
        )

        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()

    def _capture_worker(self):
        """Background thread reading audio chunks continuously."""
        while self._running:
            try:
                data = self.stream.read(2048, exception_on_overflow=False)
                raw_int16 = np.frombuffer(data, dtype=np.int16)
                if self.channels > 1:
                    raw_int16 = raw_int16.reshape(-1, self.channels)
                    mono_float = raw_int16.mean(axis=1).astype(np.float32) / 32768.0
                else:
                    mono_float = raw_int16.astype(np.float32) / 32768.0

                chunk_16k = resample_to_16k(mono_float, self.native_rate, 16000)
                self.audio_queue.put(chunk_16k)
            except Exception:
                if not self._running:
                    break
                time.sleep(0.01)

    def stop_streaming(self):
        """Stop continuous background capture."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def close(self):
        """Clean up PyAudio resources."""
        self.stop_streaming()
        if self.p:
            self.p.terminate()
            self.p = None

