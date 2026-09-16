"""Canonical entity registry shared by meeting and project graphs."""
from __future__ import annotations
import hashlib
import re


def _key(value):
    return " ".join(re.findall(r"(?iu)[a-zа-яё0-9]+", str(value or "").casefold()))


class EntityRegistry:
    def __init__(self, vocabulary=None):
        self.entities = {}
        # An alias may legitimately name several participants.  Keeping only
        # the last registration silently invents an identity, so the registry
        # stores a candidate set and requires context for disambiguation.
        self.aliases = {}
        for item in vocabulary or []:
            self.register(item.get("canonical_name") or item.get("name"), item.get("type", "unknown"), item.get("aliases", []), item.get("entity_id"))

    def register(self, name, kind="unknown", aliases=(), entity_id=None):
        canonical = _key(name)
        existing = self.aliases.get(canonical, set())
        if len(existing) == 1 and (not entity_id or entity_id in existing):
            entity = self.entities[next(iter(existing))]
            if entity.get("type") == kind or kind == "unknown":
                return entity
        entity_id = entity_id or "ENT" + hashlib.sha256(f"{kind}|{canonical}".encode()).hexdigest()[:12]
        entity = self.entities.setdefault(entity_id, {"entity_id": entity_id, "canonical_name": name, "type": kind, "aliases": []})
        for alias in [name, *aliases]:
            normalized = _key(alias)
            if normalized:
                self.aliases.setdefault(normalized, set()).add(entity_id)
                if alias != name and alias not in entity["aliases"]:
                    entity["aliases"].append(alias)
        return entity

    def resolve_candidates(self, mention, kind=None):
        ids = sorted(self.aliases.get(_key(mention), set()))
        return [self.entities[value] for value in ids
                if value in self.entities and (kind is None or self.entities[value]["type"] == kind)]

    def resolve(self, mention, kind=None, context_entity_ids=()):
        candidates = self.resolve_candidates(mention, kind)
        if len(candidates) == 1:
            return candidates[0]
        contextual = [item for item in candidates if item["entity_id"] in set(context_entity_ids)]
        return contextual[0] if len(contextual) == 1 else None

    def snapshot(self):
        return {
            "schema": "EntityRegistry", "schema_version": 2,
            "entities": sorted(self.entities.values(), key=lambda x: x["entity_id"]),
            "ambiguous_aliases": {
                alias: sorted(ids) for alias, ids in sorted(self.aliases.items()) if len(ids) > 1
            },
        }
