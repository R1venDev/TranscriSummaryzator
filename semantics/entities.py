"""Canonical entity registry shared by meeting and project graphs."""
from __future__ import annotations
import hashlib
import re


def _key(value):
    return " ".join(re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()))


class EntityRegistry:
    def __init__(self, vocabulary=None):
        self.entities = {}
        self.aliases = {}
        for item in vocabulary or []:
            self.register(item.get("canonical_name") or item.get("name"), item.get("type", "unknown"), item.get("aliases", []), item.get("entity_id"))

    def register(self, name, kind="unknown", aliases=(), entity_id=None):
        canonical = _key(name)
        existing = self.aliases.get(canonical)
        if existing:
            return self.entities[existing]
        entity_id = entity_id or "ENT" + hashlib.sha256(f"{kind}|{canonical}".encode()).hexdigest()[:12]
        entity = self.entities.setdefault(entity_id, {"entity_id": entity_id, "canonical_name": name, "type": kind, "aliases": []})
        for alias in [name, *aliases]:
            normalized = _key(alias)
            if normalized:
                self.aliases[normalized] = entity_id
                if alias != name and alias not in entity["aliases"]:
                    entity["aliases"].append(alias)
        return entity

    def resolve(self, mention, kind=None):
        entity_id = self.aliases.get(_key(mention))
        entity = self.entities.get(entity_id)
        return entity if entity and (kind is None or entity["type"] == kind) else None

    def snapshot(self):
        return {"schema": "EntityRegistry", "schema_version": 1, "entities": sorted(self.entities.values(), key=lambda x: x["entity_id"])}
