"""
Meeting Copilot - One-Click Launcher
Starts the backend FastAPI server and opens Chrome in standalone sidebar window mode.
"""

import os
import sys
import time
import subprocess
import webbrowser
import threading

def find_chrome_executable():
    """Locate Google Chrome executable on Windows."""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None

def launch_sidebar_browser():
    """Wait for server to start, then launch sidebar in app mode."""
    time.sleep(2.0)
    url = "http://127.0.0.1:8000"
    chrome_path = find_chrome_executable()

    if chrome_path:
        print(f"[Launcher] Opening sidebar window in Chrome: {chrome_path}")
        # Launch Chrome as a slim vertical app window next to meetings
        args = [
            chrome_path,
            f"--app={url}",
            "--window-size=410,820",
        ]
        try:
            subprocess.Popen(args)
            return
        except Exception as e:
            print(f"[Launcher] Failed to launch Chrome app mode: {e}")

    # Fallback to default browser
    print(f"[Launcher] Opening {url} in default web browser...")
    webbrowser.open(url)

def main():
    print("=" * 65)
    print("           STARTING MEETING COPILOT")
    print("=" * 65)
    print("Backend: http://127.0.0.1:8000")
    print("Models : faster-whisper (GPU) + Gemma 3 1B (Ollama)")
    print("-" * 65)

    # Launch browser sidebar in background thread
    threading.Thread(target=launch_sidebar_browser, daemon=True).start()

    # Start FastAPI server via uvicorn
    try:
        import uvicorn
        from config import HOST, PORT
        uvicorn.run("app:app", host=HOST, port=PORT, reload=False, log_level="info")
    except ImportError:
        print("[ERROR] FastAPI or Uvicorn is not installed.")
        print("Run: pip install fastapi \"uvicorn[standard]\"")
        sys.exit(1)

if __name__ == "__main__":
    main()

