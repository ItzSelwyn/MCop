"""
Ollama Gemma Worker for Meeting Copilot.
Handles:
1. Name mention alert verification ("Is this directed at me?")
2. Rolling meeting summary generation (Key Points, Decisions, Action Items)
"""

import json
import re
import time
from typing import Dict, Any, Optional

try:
    import ollama
except ImportError:
    ollama = None

from config import OLLAMA_MODEL, OLLAMA_HOST, DEFAULT_USER_NAME, DEFAULT_ALIASES, EXACT_ALIASES


def extract_json_safely(raw_text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON object from raw LLM output."""
    if not raw_text:
        return None
    try:
        return json.loads(raw_text)
    except Exception:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


class OllamaWorker:
    def __init__(self, model_name: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model_name = model_name
        self.host = host
        self.client = None
        if ollama is not None:
            try:
                self.client = ollama.Client(host=self.host)
            except Exception as e:
                print(f"[OllamaWorker] Could not initialize client: {e}")

    def is_available(self) -> bool:
        """Check if Ollama service is reachable."""
        if not self.client:
            return False
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def verify_mention(
        self,
        sentence: str,
        context: str,
        user_name: str = DEFAULT_USER_NAME
    ) -> Dict[str, Any]:
        """
        Ask Gemma if the detected sentence is directed at user_name.
        Returns:
            {"directed": bool, "kind": str, "reason": str, "quote": str}
        """
        clean_s = sentence.lower()
        all_aliases = [user_name.lower()] + [a.lower() for a in DEFAULT_ALIASES + EXACT_ALIASES]

        # Guardrail: If none of the user's aliases appear as words in the sentence, reject immediately
        words = re.findall(r"\b\w+\b", clean_s)
        has_alias = any(alias in words for alias in all_aliases)
        if not has_alias:
            return {
                "directed": False,
                "kind": "none",
                "reason": f"'{user_name}' is not in this sentence",
                "quote": sentence,
            }

        fallback_response = {
            "directed": True,
            "kind": "mention",
            "reason": f"Direct address to {user_name} detected",
            "quote": sentence,
        }

        if not self.client:
            return fallback_response

        prompt = (
            f"You are a meeting alert filter for {user_name}.\n"
            f"Analyze if the speaker in the sentence below is DIRECTLY addressing {user_name} (e.g. asking {user_name} a question, assigning a task, or calling on them).\n\n"
            f"Recent Context:\n{context or 'No prior context.'}\n\n"
            f"Sentence to evaluate:\n\"{sentence}\"\n\n"
            "Rules:\n"
            f"- If the speaker is speaking directly TO {user_name}, return directed=true.\n"
            f"- If the speaker is talking ABOUT someone else, casual chatter, or {user_name} is not being addressed, return directed=false.\n\n"
            "Respond ONLY with valid JSON:\n"
            "{\n"
            '  "directed": true,\n'
            '  "kind": "question",\n'
            '  "reason": "short explanation"\n'
            "}"
        )

        try:
            resp = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.1, "num_predict": 120},
            )
            raw = resp.get("message", {}).get("content", "")
            data = extract_json_safely(raw)
            if data and "directed" in data:
                data["quote"] = sentence
                return data
        except Exception as e:
            print(f"[OllamaWorker] verify_mention error: {e}")

        return fallback_response

    def generate_summary(self, transcript_text: str) -> Dict[str, Any]:
        """
        Generate structured meeting summary: Key Points, Decisions, Action Items.
        """
        empty_summary = {
            "key_points": [],
            "decisions": [],
            "action_items": [],
            "timestamp": time.strftime("%H:%M:%S")
        }

        if not transcript_text or not self.client:
            return empty_summary

        prompt = (
            "You are an executive meeting assistant. Analyze the following meeting transcript. "
            "Extract concise key points, confirmed decisions, and assigned action items.\n\n"
            f"Meeting Transcript:\n{transcript_text}\n\n"
            "Respond ONLY with valid JSON in this exact structure:\n"
            "{\n"
            '  "key_points": ["point 1", "point 2"],\n'
            '  "decisions": ["decision 1"],\n'
            '  "action_items": [\n'
            '    {"task": "description", "owner": "name or Team"}\n'
            "  ]\n"
            "}"
        )

        try:
            resp = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.2, "num_predict": 250},
            )
            raw = resp.get("message", {}).get("content", "")
            data = extract_json_safely(raw)
            if data:
                data["timestamp"] = time.strftime("%H:%M:%S")
                data.setdefault("key_points", [])
                data.setdefault("decisions", [])
                data.setdefault("action_items", [])
                return data
        except Exception as e:
            print(f"[OllamaWorker] generate_summary error: {e}")

        return empty_summary
