"""Bounded, local scene hints. Positions refer to the image, never to distance or safety."""

from dataclasses import dataclass
from math import isfinite
import os
import re

from .history import normalized

PRIORITY = {"chair": 5, "bench": 5, "couch": 4, "dining table": 4, "bicycle": 4,
            "suitcase": 4, "backpack": 3, "person": 2, "text": 3}


def image_region(box, width):
    if len(box) != 4 or width <= 0 or any(not isfinite(coord) for coord in box):
        return None
    x = (box[0] + box[2]) / (2 * width)
    return "left" if x < .33 else "right" if x > .67 else "center"


def scene_items(analysis, confidence):
    candidates = {}
    width = analysis["width"]
    for obj in sorted(analysis["objects"], key=lambda obj: obj.get("confidence", 0), reverse=True):
        label = obj["name"]
        if label in PRIORITY and obj["confidence"] >= confidence:
            position = image_region(obj.get("bbox", []), width)
            if position:
                candidates.setdefault(label, position)
    if any(len(word["text"].strip()) >= 3 and word.get("confidence", 0) >= max(confidence, .75)
           for word in analysis.get("ocr_results", [])):
        candidates["text"] = None
    return candidates


@dataclass(frozen=True)
class SceneSettings:
    confidence: float = .6
    confirmations: int = 2
    cooldown: float = 18

    @classmethod
    def from_env(cls):
        settings = cls(float(os.getenv("SCENE_CONFIDENCE", ".6")), int(os.getenv("SCENE_CONFIRMATIONS", "2")),
                       float(os.getenv("SCENE_COOLDOWN_SECONDS", "18")))
        if not (.0 <= settings.confidence <= 1 and 2 <= settings.confirmations <= 10
                and 3 <= settings.cooldown <= 300):
            raise ValueError("Invalid scene configuration: confidence 0-1, confirmations 2-10, cooldown 3-300 seconds")
        return settings


class SceneTracker:
    def __init__(self, settings):
        self.settings = settings
        self.previous = {}
        self.present = {}
        self.pending = set()
        self.last_at = -float("inf")
        self.event = None
        self.sequence = 0

    def observe(self, analysis, now):
        if analysis.get("quality") == "blurry" or analysis.get("analysis_errors"):
            self.previous.clear()
            return
        items = scene_items(analysis, self.settings.confidence)
        consecutive = self.settings.confirmations
        self.previous = {label: min(consecutive, self.previous.get(label, 0) + 1) for label in items}
        for label in tuple(self.present):
            self.present[label] = 0 if label in items else self.present[label] + 1
            if self.present[label] >= consecutive:
                del self.present[label]
                self.pending.discard(label)
        for label, count in self.previous.items():
            if count == consecutive and label not in self.present:
                self.present[label] = 0
                self.pending.add(label)
        eligible = [label for label in self.pending if self.previous.get(label) == consecutive]
        if not eligible or now - self.last_at < self.settings.cooldown:
            return
        label = max(eligible, key=lambda key: (items[key] == "center", PRIORITY[key], key))
        self.pending.difference_update(eligible)
        self.sequence += 1
        self.last_at = now
        self.event = {"sequence": self.sequence, "label": label, "region": items[label]}

    def latest(self, now):
        return self.event if now - self.last_at <= 6 else None


def safety_limit(question, language):
    phrase = normalized(question)
    if not re.search(r"\b(?:safe to|can i cross|can i walk (?:forward|through)|how far|is the (?:path|way) clear|"
                     r"puis-je traverser|est-ce (?:sur|sans danger)|a quelle distance|voie est-elle libre)\b", phrase):
        return None
    return ("La cam\u00e9ra ne permet pas d'estimer la distance ni de confirmer que le passage est s\u00fbr. "
            "Utilisez vos aides habituelles au d\u00e9placement." if language == "fr" else
            "I can't estimate distance or confirm safe passage from this camera. Use your usual mobility aids.")
