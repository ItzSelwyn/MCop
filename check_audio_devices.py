"""
Audio Devices Diagnostic Tool for Meeting Copilot.
Lists all Windows WASAPI audio devices, highlights Loopback devices,
and tests opening the default loopback stream.
"""

import sys
from audio_capture import setup_cuda_dll_paths
setup_cuda_dll_paths()

def main():
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        print("\n[ERROR] PyAudioWPatch is not installed.")
        print("Please activate your virtual environment and run:")
        print("    pip install PyAudioWPatch\n")
        sys.exit(1)

    p = pyaudio.PyAudio()

    print("=" * 65)
    print("      MEETING COPILOT - AUDIO DEVICE DIAGNOSTIC")
    print("=" * 65)

    # 1. WASAPI Check
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        print(f"[OK] WASAPI Host API found (Index: {wasapi_info['index']}, Name: {wasapi_info['name']})")
    except OSError:
        print("[ERROR] WASAPI Host API is not available on this machine.")
        p.terminate()
        sys.exit(1)

    # 2. Default Output Device
    try:
        default_speakers = p.get_default_output_device_info()
        print(f"[OK] Default Speakers: '{default_speakers['name']}' (Index: {default_speakers['index']})")
    except Exception as e:
        print(f"[WARNING] Could not determine default output device: {e}")
        default_speakers = None

    # 3. Default Loopback Device
    print("-" * 65)
    try:
        default_loopback = p.get_default_wasapi_loopback()
        if default_loopback:
            print(f"[OK] Default WASAPI Loopback Device: '{default_loopback['name']}'")
            print(f"     Device Index : {default_loopback['index']}")
            print(f"     Sample Rate  : {int(default_loopback['defaultSampleRate'])} Hz")
            print(f"     Channels     : {default_loopback['maxInputChannels']}")
        else:
            print("[WARNING] get_default_wasapi_loopback() returned None.")
    except Exception as e:
        print(f"[WARNING] Failed to query default WASAPI loopback: {e}")
        default_loopback = None

    # 4. List All Available Loopback Devices
    print("-" * 65)
    print("Available WASAPI Loopback Capture Devices:")
    loopback_count = 0
    for dev in p.get_loopback_device_info_generator():
        loopback_count += 1
        marker = " <== [DEFAULT]" if default_loopback and dev['index'] == default_loopback['index'] else ""
        print(f"  [{dev['index']:2d}] {dev['name']}{marker}")
        print(f"       Rate: {int(dev['defaultSampleRate'])} Hz | Channels: {dev['maxInputChannels']}")

    if loopback_count == 0:
        print("\n[ALERT] No loopback devices found! Make sure your headphones/speakers are plugged in and enabled in Windows Sound Settings.")
        p.terminate()
        sys.exit(1)

    # 5. Sanity Test: Open and capture 1 second of audio
    print("-" * 65)
    test_device = default_loopback
    if not test_device:
        # Fallback to first loopback device
        for dev in p.get_loopback_device_info_generator():
            test_device = dev
            break

    print(f"Testing 1-second capture from: '{test_device['name']}'...")
    try:
        sample_rate = int(test_device["defaultSampleRate"])
        channels = int(test_device["maxInputChannels"])
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=test_device["index"],
            frames_per_buffer=1024,
        )
        
        frames = []
        for _ in range(int(sample_rate / 1024)):
            frames.append(stream.read(1024, exception_on_overflow=False))
            
        stream.stop_stream()
        stream.close()
        print("[SUCCESS] Successfully captured 1 second of loopback audio without errors!")
        print("          Your audio loopback setup is 100% READY.")
    except Exception as e:
        print(f"[ERROR] Failed to capture test audio: {e}")
    finally:
        p.terminate()

    print("=" * 65)

if __name__ == "__main__":
    main()

