"""Generate domain corruptions for the small meeting verifier."""
from __future__ import annotations
import re


def corrupt(claim):
    text = str(claim.get("statement") or "")
    values = []
    speakers = claim.get("speaker_refs", [])
    if speakers:
        values.append({"type": "speaker_swap", "supported": text, "corrupted": text.replace(speakers[0], "@OTHER")})
    number = re.search(r"\d+(?:[.,]\d+)?", text)
    if number:
        replacement = str(float(number.group().replace(",", ".")) + 1).rstrip("0").rstrip(".")
        values.append({"type": "number_swap", "supported": text, "corrupted": text[:number.start()] + replacement + text[number.end():]})
    if re.search(r"(?iu)\b(?:может|вероятно|предлагается)\b", text):
        values.append({"type": "modality_upgrade", "supported": text, "corrupted": re.sub(r"(?iu)\b(?:может|вероятно|предлагается)\b", "решено", text, count=1)})
    if re.search(r"(?iu)\bне\b", text):
        values.append({"type": "negation_deletion", "supported": text, "corrupted": re.sub(r"(?iu)\bне\s+", "", text, count=1)})
    return values
