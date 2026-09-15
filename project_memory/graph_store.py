"""Durable cross-meeting ProjectGraph event store with semantic lineage."""
from __future__ import annotations
import fcntl, hashlib, json, os, tempfile, time
from pathlib import Path

def _family(claim):
    sig = claim.get("semantic_signature", {})
    stable = {k: sig.get(k) for k in ("subject", "predicate", "object", "scope", "conditions")}
    if not any(stable.values()): stable = {"fallback": " ".join(str(claim.get("statement", "")).casefold().split())}
    return "LIN" + hashlib.sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]

def empty(project):
    return {"schema": "ProjectGraphSchema", "schema_version": 2, "project": project, "revision": 0, "entities": {}, "propositions": {}, "decisions": {}, "tasks": {}, "experiments": {}, "threads": {}, "meeting_events": [], "lineage": {}, "history": []}

def apply_meeting(project_graph, meeting_graph):
    graph = json.loads(json.dumps(project_graph)); meeting_id = meeting_graph["meeting_id"]
    replacing = meeting_id in {x["meeting_id"] for x in graph["meeting_events"]}
    if replacing:
        for bucket in ("propositions", "decisions", "tasks", "experiments", "threads"):
            graph[bucket] = {key: value for key, value in graph[bucket].items() if value.get("meeting_id") != meeting_id}
        graph["meeting_events"] = [x for x in graph["meeting_events"] if x["meeting_id"] != meeting_id]
        graph["lineage"] = {family: [key for key in keys if key in graph["propositions"]]
                            for family, keys in graph["lineage"].items()}
    changes = []
    for claim in meeting_graph.get("claims", []):
        family = _family(claim); prior_ids = graph["lineage"].get(family, []); prior = graph["propositions"].get(prior_ids[-1]) if prior_ids else None
        status = "NEW" if not prior else "CHANGED" if prior.get("polarity") != claim.get("polarity") or prior.get("quantities") != claim.get("quantities") else "CONFIRMS"
        entry = {**claim, "meeting_id": meeting_id, "lineage_id": family, "project_revision": graph["revision"] + 1, "recorded_at": time.time()}
        project_key = meeting_id + ":" + claim["proposition_id"]
        graph["propositions"][project_key] = entry; graph["lineage"].setdefault(family, []).append(project_key)
        changes.append({"status": status, "lineage_id": family, "proposition_id": claim["proposition_id"], "project_key": project_key, "previous_proposition_id": prior_ids[-1] if prior_ids else None})
        for entity in claim.get("entities", []): graph["entities"][entity["entity_id"]] = entity
    for source, target, key in (("decision_states", "decisions", "decision_id"), ("task_states", "tasks", "task_id"), ("experiment_states", "experiments", "experiment_id"), ("threads", "threads", "thread_id")):
        for item in meeting_graph.get(source, []): graph[target][meeting_id + ":" + item[key]] = {**item, "meeting_id": meeting_id}
    graph["meeting_events"].append({"meeting_id": meeting_id, "generation_id": meeting_graph.get("generation_id"), "delta": changes, "provenance": meeting_graph.get("provenance", {})}); graph["revision"] += 1
    graph["history"].append({"revision": graph["revision"], "meeting_id": meeting_id,
                             "generation_id": meeting_graph.get("generation_id"), "transition": "regenerated" if replacing else "new_meeting"})
    graph["project_graph_id"] = "PG" + hashlib.sha256(json.dumps(graph["history"], sort_keys=True).encode()).hexdigest()[:16]
    return graph, {"schema_version": 2, "meeting_id": meeting_id, "transition": "regenerated" if replacing else "new_meeting", "changes": changes}

class ProjectGraphStore:
    def __init__(self, root, project):
        safe = "".join(x if x.isalnum() or x in "-_" else "_" for x in project) or "default"; self.directory = Path(root) / safe; self.path = self.directory / "project_graph.json"
    def load(self): return json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else empty(self.directory.name)
    def publish(self, meeting_graph):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".project_graph.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            graph, delta = apply_meeting(self.load(), meeting_graph)
            fd, name = tempfile.mkstemp(prefix="project-graph-", dir=self.directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(graph, stream, ensure_ascii=False, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
                os.replace(name, self.path)
            finally:
                if os.path.exists(name): os.unlink(name)
            return graph, delta
