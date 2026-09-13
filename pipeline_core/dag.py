"""Small declarative DAG with independently cacheable stages."""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable


@dataclass(frozen=True)
class Stage:
    name: str
    version: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    runner: Callable | None = None


class DAG:
    def __init__(self, stages):
        self.stages = {x.name: x for x in stages}
        for stage in stages:
            missing = set(stage.dependencies) - self.stages.keys()
            if missing:
                raise ValueError(f"{stage.name}: unknown dependencies {sorted(missing)}")

    def order(self, target):
        result, visiting, visited = [], set(), set()
        def visit(name):
            if name in visiting:
                raise ValueError("cycle in pipeline DAG")
            if name in visited:
                return
            visiting.add(name)
            for dependency in self.stages[name].dependencies:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            result.append(self.stages[name])
        visit(target)
        return result


MEETING_DAG = DAG([
    Stage("audio", "v1", ("source",), ("canonical_audio",)),
    Stage("diarization", "v2", ("canonical_audio",), ("speaker_segments",), ("audio",)),
    Stage("asr", "v2", ("canonical_audio",), ("words", "asr_alternatives"), ("audio",)),
    Stage("evidence", "v2", ("words", "speaker_segments"), ("evidence_ledger",), ("asr", "diarization")),
    Stage("claims", "v1", ("evidence_ledger",), ("claims",), ("evidence",)),
    Stage("episodes", "v1", ("claims",), ("episodes", "threads"), ("claims",)),
    Stage("claim_graph", "v1", ("claims", "episodes"), ("relations", "meeting_state"), ("episodes",)),
    Stage("project_state", "v1", ("meeting_state",), ("project_state", "delta"), ("claim_graph",)),
    Stage("summary_plan", "v2", ("meeting_state",), ("summary_plan",), ("claim_graph",)),
    Stage("rendering", "v2", ("summary_plan",), ("verified_views",), ("summary_plan",)),
])
