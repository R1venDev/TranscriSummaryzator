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
    input_schema: str = "Any/v1"
    output_schema: str = "Any/v1"
    model_digest: str | None = None
    retry_policy: str = "bounded"
    failure_policy: str = "fail_closed"
    degradation_policy: str = "none"
    metrics: tuple[str, ...] = ()


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


def _s(name, version, inputs, outputs, dependencies=(), ins="Any/v1", outs="Any/v1", failure="fail_closed", degradation="none", metrics=()):
    return Stage(name, version, inputs, outputs, dependencies, input_schema=ins, output_schema=outs, failure_policy=failure, degradation_policy=degradation, metrics=metrics)

MEETING_DAG = DAG([
    _s("01_audio", "v2", ("source",), ("canonical_audio",), outs="Audio/v2"),
    _s("02_diarization_primary", "v3", ("canonical_audio",), ("primary_segments",), ("01_audio",), "Audio/v2", "SpeakerSegments/v3", metrics=("DER",)),
    _s("03_diarization_secondary", "v3", ("canonical_audio",), ("secondary_segments",), ("01_audio",), "Audio/v2", "SpeakerSegments/v3", metrics=("DER",)),
    _s("04_speaker_consensus", "v3", ("primary_segments", "secondary_segments"), ("speaker_segments",), ("02_diarization_primary", "03_diarization_secondary"), outs="SpeakerConsensus/v3", metrics=("DER", "JER")),
    _s("05_voice_identity", "v3", ("speaker_segments",), ("identified_segments",), ("04_speaker_consensus",), outs="SpeakerIdentity/v3", metrics=("ECE", "Brier")),
    _s("06_asr", "v3", ("canonical_audio",), ("words", "asr_lattice"), ("01_audio",), outs="WordEvidence/v3", metrics=("critical_WER",)),
    _s("07_evidence_build", "v3", ("words", "identified_segments"), ("evidence_spans",), ("05_voice_identity", "06_asr"), outs="EvidenceSpan/v3"),
    _s("08_evidence_repair", "v3", ("evidence_spans",), ("repaired_evidence",), ("07_evidence_build",), outs="EvidenceSpan/v3", failure="risk_based", degradation="abstain"),
    _s("09_proposition_extract", "v4", ("repaired_evidence",), ("propositions",), ("08_evidence_repair",), outs="PropositionSchema/v4"),
    _s("10_dialogue_act", "v2", ("propositions",), ("dialogue_events",), ("09_proposition_extract",), outs="DialogueActSchema/v2"),
    _s("11_relation_resolve", "v3", ("propositions", "dialogue_events"), ("relations",), ("10_dialogue_act",), outs="RelationSchema/v3", metrics=("relation_F1",)),
    _s("12_state_reduce", "v4", ("propositions", "dialogue_events", "relations"), ("states",), ("11_relation_resolve",), outs="StateMachines/v4"),
    _s("13_episode_segment", "v2", ("propositions", "dialogue_events"), ("episodes",), ("12_state_reduce",), outs="Episodes/v2", metrics=("boundary_F1",)),
    _s("14_thread_resolve", "v5", ("episodes", "relations"), ("meeting_graph",), ("13_episode_segment",), outs="MeetingGraphSchema/v7", metrics=("thread_score",)),
    _s("15_project_delta", "v2", ("meeting_graph",), ("project_graph", "delta"), ("14_thread_resolve",), outs="ProjectGraphSchema/v2"),
    _s("16_view_plan", "v5", ("meeting_graph", "project_graph"), ("view_plans",), ("15_project_delta",), outs="SummaryPlanSchema/v5"),
    _s("17_realize", "v6", ("view_plans",), ("public_items",), ("16_view_plan",), outs="PublicItemSchema/v4"),
    _s("18_verify", "v6", ("public_items", "meeting_graph"), ("verified_public_items",), ("17_realize",), outs="PublicationAuditSchema/v5", degradation="abstain", metrics=("public_precision", "condition_preservation", "orphan_public_items", "status_upgrades", "duplicates", "chronology_inversions", "unknown_semantic_checks", "rendered_node_coverage")),
    _s("19_publish", "v4", ("verified_public_items",), ("published_outputs",), ("18_verify",), outs="PublishedMeeting/v4"),
])
