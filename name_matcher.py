"""
Fast Name & Group Question Matcher for Meeting Copilot.
Handles:
1. Precision matching for your name / phonetic aliases (zero false positives).
2. Group question detection ("did you all...", "can anyone...", "does anyone...").
"""

import re
from typing import List, Tuple, Optional, Dict, Any
from rapidfuzz import fuzz
from config import (
    DEFAULT_ALIASES,
    EXACT_ALIASES,
    FUZZY_MATCH_THRESHOLD,
    ALERT_ON_GROUP_QUESTIONS,
)

# Regex patterns for questions/requests addressed to the room, team, or everyone
GROUP_QUESTION_PATTERNS = [
    # "Did you all", "Were you all", "Are you all", "Can you all", "Have you all", "Do you all"
    r"\b(?:did|were|are|have|do|can|could|will|would)\s+(?:you\s+all|y\'all|everyone|everybody)\b",
    # "Does anyone", "Can anyone", "Could anyone", "Has anyone", "Is anyone", "Can somebody"
    r"\b(?:does|can|could|has|have|is|will|would)\s+(?:anyone|anybody|someone|somebody)\b",
    # "What do you all think", "What does everyone think", "How do you all feel"
    r"\b(?:what|how)\s+(?:do|does|are|about)\s+(?:you\s+all|y\'all|everyone|the\s+team|folks|guys)\b",
    # "Any thoughts", "Any questions", "Any concerns", "Any feedback", "Any volunteers"
    r"\bany\s+(?:thoughts|questions|concerns|feedback|volunteers|updates|input|blockers)\b",
    # Direct team address with a question: "Team, can we...", "Guys, does anyone..."
    r"\b(?:team|folks|guys|everyone)\b[^\.\?!]*(?:\?|\b(?:can\s+we|can\s+you|does\s+anyone|has\s+anyone|are\s+we)\b)",
    # "Who can take...", "Who wants to..."
    r"\b(?:who\s+wants\s+to|who\s+can|who\s+is\s+going\s+to|who\s+would\s+like\s+to)\b",
    # "Let me ask you all", "Let me ask everyone"
    r"\blet\s+me\s+ask\s+(?:you\s+all|y\'all|everyone|the\s+team)\b",
    # "Are there any questions/concerns"
    r"\bare\s+there\s+any\s+(?:questions|concerns|thoughts)\b",
]

_GROUP_REGEX = re.compile("|".join(GROUP_QUESTION_PATTERNS), re.IGNORECASE)


class NameMatcher:
    def __init__(
        self,
        aliases: Optional[List[str]] = None,
        exact_aliases: Optional[List[str]] = None,
        threshold: int = FUZZY_MATCH_THRESHOLD,
        detect_group: bool = ALERT_ON_GROUP_QUESTIONS,
    ):
        self.aliases = [a.lower().strip() for a in (aliases or DEFAULT_ALIASES) if a.strip()]
        self.exact_aliases = [a.lower().strip() for a in (exact_aliases or EXACT_ALIASES) if a.strip()]
        self.threshold = threshold
        self.detect_group = detect_group

    def check_direct_name(self, text: str) -> Tuple[bool, Optional[str], Optional[str], float]:
        """
        Check if any word in `text` matches your name or phonetic aliases.
        Returns:
            (is_match, detected_word, matched_alias, score)
        """
        if not text:
            return False, None, None, 0.0

        # Clean punctuation and tokenize words
        clean_text = re.sub(r"[^\w\s]", " ", text.lower())
        words = clean_text.split()

        # 1. Exact match for short aliases (e.g. "sel")
        for word in words:
            if word in self.exact_aliases:
                return True, word, word, 100.0

        # 2. Fuzzy match for longer aliases (length >= 4, e.g. "selwyn", "selvin")
        best_score = 0.0
        best_word = None
        best_alias = None

        for word in words:
            if len(word) < 4:
                continue
            for alias in self.aliases:
                # Direct equality
                if word == alias:
                    return True, word, alias, 100.0

                score = fuzz.ratio(word, alias)
                if score > best_score:
                    best_score = score
                    best_word = word
                    best_alias = alias

        if best_score >= self.threshold:
            return True, best_word, best_alias, float(best_score)

        return False, None, None, float(best_score)

    def check_group_question(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if `text` is a question or request directed at the group/everyone.
        Returns:
            (is_group_question, matched_phrase)
        """
        if not self.detect_group or not text:
            return False, None

        match = _GROUP_REGEX.search(text)
        if match:
            return True, match.group(0).strip()

        return False, None

    def check(self, text: str) -> Dict[str, Any]:
        """
        Unified check for both direct mentions and group questions.
        Returns:
            {
                "is_alert": bool,
                "type": "direct" | "group" | "none",
                "detected_word": str | None,
                "alias": str | None,
                "matched_phrase": str | None,
                "score": float
            }
        """
        # 1. Direct name check (Highest priority)
        is_direct, word, alias, score = self.check_direct_name(text)
        if is_direct:
            return {
                "is_alert": True,
                "type": "direct",
                "detected_word": word,
                "alias": alias,
                "matched_phrase": word,
                "score": score,
            }

        # 2. Group question check
        is_group, phrase = self.check_group_question(text)
        if is_group:
            return {
                "is_alert": True,
                "type": "group",
                "detected_word": phrase,
                "alias": None,
                "matched_phrase": phrase,
                "score": 100.0,
            }

        return {
            "is_alert": False,
            "type": "none",
            "detected_word": None,
            "alias": None,
            "matched_phrase": None,
            "score": 0.0,
        }
