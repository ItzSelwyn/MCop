"""
Thread-safe Rolling Transcript Buffer for Meeting Copilot.
Stores recent conversation chunks with timestamps and provides context windows.
"""

import time
import datetime
import threading
from typing import List, Dict, Any, Optional

class TranscriptBuffer:
    def __init__(self, max_history_seconds: int = 1800):  # 30 minutes in memory
        self.max_history_seconds = max_history_seconds
        self.segments: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._counter = 0

    def add_segment(self, text: str, infer_time: float = 0.0) -> Dict[str, Any]:
        """Add a newly transcribed segment to the buffer."""
        now = time.time()
        time_str = datetime.datetime.fromtimestamp(now).strftime("%H:%M:%S")

        with self._lock:
            self._counter += 1
            entry = {
                "id": self._counter,
                "timestamp": time_str,
                "epoch": now,
                "text": text.strip(),
                "infer_time": round(infer_time, 2)
            }
            self.segments.append(entry)
            self._prune_locked(now)
            return entry

    def _prune_locked(self, current_time: float):
        """Remove entries older than max_history_seconds."""
        cutoff = current_time - self.max_history_seconds
        self.segments = [s for s in self.segments if s["epoch"] >= cutoff]

    def get_recent_segments(self, seconds: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get segments from the last N seconds (or all if None)."""
        now = time.time()
        with self._lock:
            if seconds is None:
                return list(self.segments)
            cutoff = now - seconds
            return [s for s in self.segments if s["epoch"] >= cutoff]

    def get_context_text(self, seconds: int = 35) -> str:
        """Get context string from the last N seconds for prompt grounding."""
        recent = self.get_recent_segments(seconds=seconds)
        if not recent:
            return ""
        lines = [f"[{s['timestamp']}] {s['text']}" for s in recent]
        return "\n".join(lines)

    def get_full_text(self, seconds: int = 300) -> str:
        """Get text from the last N seconds (e.g. 5 minutes) for meeting summarization."""
        return self.get_context_text(seconds=seconds)

    def export_markdown(self) -> str:
        """Export the current meeting transcript as formatted Markdown."""
        with self._lock:
            if not self.segments:
                return "# Meeting Transcript\n\nNo audio captured yet."
            
            start_time = self.segments[0]["timestamp"]
            end_time = self.segments[-1]["timestamp"]
            lines = [
                f"# Meeting Transcript",
                f"**Session**: {start_time} - {end_time}",
                f"**Total Segments**: {len(self.segments)}\n",
                "| Time | Transcript |",
                "| :--- | :--- |"
            ]
            for s in self.segments:
                lines.append(f"| `{s['timestamp']}` | {s['text']} |")
            return "\n".join(lines)

