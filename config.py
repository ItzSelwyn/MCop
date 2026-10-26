"""
Meeting Copilot - Configuration Settings
"""

import os
from typing import List

# Target user information for alert filtering
DEFAULT_USER_NAME: str = "Selwyn"

# Phonetic and common Whisper misrecognition aliases for your name (length >= 4)
DEFAULT_ALIASES: List[str] = [
    "selwyn",
    "selvin",
    "selwin",
    "salwin",
]

# Short aliases that must be an EXACT word match (prevents fuzzy collisions with common words)
EXACT_ALIASES: List[str] = [
    "sel",
]

# RapidFuzz match threshold (0 - 100) for name aliases
FUZZY_MATCH_THRESHOLD: int = 82

# Alert on questions directed at the whole room / team / everyone
ALERT_ON_GROUP_QUESTIONS: bool = True

# Audio & Whisper Settings
CHUNK_DURATION_SEC: float = 5.0       # Duration of rolling audio chunks
SILENCE_RMS_THRESHOLD: float = 0.003  # Below this, chunk is skipped as silence
WHISPER_MODEL: str = "small"          # tiny, base, small, medium
WHISPER_DEVICE: str = "cuda"          # cuda (falls back to cpu if unavailable)
WHISPER_COMPUTE: str = "float16"      # float16 or int8_float16 for GPU, int8 for CPU
WHISPER_LANGUAGE: str = "en"

# Ollama / Gemma Settings
OLLAMA_HOST: str = "http://127.0.0.1:11434"
OLLAMA_MODEL: str = "gemma3:1b"       # Model pulled by user
SUMMARY_INTERVAL_SEC: int = 60        # Seconds between rolling summaries
CONTEXT_WINDOW_SEC: int = 35          # Context duration (seconds) sent to Gemma for alert verification

# Server Settings
HOST: str = "127.0.0.1"
PORT: int = 8000
