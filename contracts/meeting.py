"""Strict, versioned contracts for evidence-backed meeting state."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
from semantics.ontology import ClaimKind, ClaimLifecycle, DecisionStatus, QuestionStatus, RelationKind, TaskStatus


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConfidenceVector(StrictModel):
    recognition: Optional[float] = Field(None, ge=0, le=1)
    speaker_identity: Optional[float] = Field(None, ge=0, le=1)
    semantic_support: Optional[float] = Field(None, ge=0, le=1)
    relation_support: Optional[float] = Field(None, ge=0, le=1)
    modality: Optional[float] = Field(None, ge=0, le=1)
    quantity: Optional[float] = Field(None, ge=0, le=1)


class RiskVector(StrictModel):
    recognition: float = Field(0, ge=0, le=1)
    speaker: float = Field(0, ge=0, le=1)
    number: float = Field(0, ge=0, le=1)
    negation: float = Field(0, ge=0, le=1)
    modality: float = Field(0, ge=0, le=1)
    relation: float = Field(0, ge=0, le=1)
    task_assignment: float = Field(0, ge=0, le=1)


class NormalizedQuantity(StrictModel):
    value: float
    unit: Optional[str] = None
    operator: Literal["exact", "approx", "min", "max", "range"] = "exact"
    direction: Optional[Literal["increase", "decrease", "neutral"]] = None


class Quantity(StrictModel):
    quantity_id: str
    raw_text: str
    normalized: NormalizedQuantity
    entity: Optional[str] = None
    evidence_ids: list[str] = Field(min_length=1)
    status: Literal["accepted", "disputed", "unknown"] = "accepted"


class TimeExpression(StrictModel):
    raw_text: str
    kind: str
    date_resolution: Optional[str] = None
    time: Optional[str] = None
    timezone: Optional[str] = None
    certainty: Literal["asserted", "proposed", "tentative", "disputed"] = "asserted"


class Condition(StrictModel):
    predicate: str
    effect: Optional[str] = None
    evidence_ids: list[str] = Field(min_length=1)


class Claim(StrictModel):
    claim_id: str
    kind: ClaimKind
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    episode_id: Optional[str] = None
    thread_id: Optional[str] = None
    speaker_refs: list[str] = Field(default_factory=list)
    modality: Literal["asserted", "tentative", "proposed", "committed", "question"] = "asserted"
    lifecycle: ClaimLifecycle = ClaimLifecycle.ACTIVE
    confidence: ConfidenceVector = Field(default_factory=ConfidenceVector)
    risk: RiskVector = Field(default_factory=RiskVector)
    quantities: list[Quantity] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)


class Relation(StrictModel):
    relation_id: str
    type: RelationKind
    source_claim_id: str
    target_claim_id: str
    evidence_ids: list[str] = Field(min_length=1)
    confidence: ConfidenceVector = Field(default_factory=ConfidenceVector)


class DialogueEpisode(StrictModel):
    episode_id: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    topic: str
    initiating_event_ids: list[str] = Field(default_factory=list)
    claim_ids: list[str] = Field(default_factory=list)
    question_ids: list[str] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    outcome_claim_ids: list[str] = Field(default_factory=list)
    open_threads: list[str] = Field(default_factory=list)


class QuestionState(StrictModel):
    question_id: str
    claim_id: str
    requested_slots: list[str] = Field(default_factory=list)
    answered_slots: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    answer_claim_ids: list[str] = Field(default_factory=list)
    status: QuestionStatus = QuestionStatus.UNANSWERED


class TaskState(StrictModel):
    task_id: str
    description: str
    proposed_by: Optional[str] = None
    assignee: Optional[str] = None
    assignment_evidence_ids: list[str] = Field(default_factory=list)
    acceptance_evidence_ids: list[str] = Field(default_factory=list)
    commitment_strength: Literal["none", "tentative", "explicit"] = "none"
    deadline: Optional[TimeExpression] = None
    conditions: list[Condition] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.IDEA


class DecisionState(StrictModel):
    decision_id: str
    proposal_claim_ids: list[str] = Field(default_factory=list)
    acceptance_claim_ids: list[str] = Field(default_factory=list)
    decision_makers: list[str] = Field(default_factory=list)
    scope: str
    conditions: list[Condition] = Field(default_factory=list)
    status: DecisionStatus = DecisionStatus.CANDIDATE


class SentencePlan(StrictModel):
    sentence_id: str
    episode_id: Optional[str] = None
    claim_ids: list[str] = Field(min_length=1)
    relation_ids: list[str] = Field(default_factory=list)
    intent: str
    allowed_numbers: list[str] = Field(default_factory=list)
    allowed_speakers: list[str] = Field(default_factory=list)
    max_sentences: int = Field(1, ge=1, le=3)


class ParagraphPlan(StrictModel):
    paragraph_id: str
    episode_ids: list[str] = Field(min_length=1)
    claim_ids: list[str] = Field(min_length=1)
    relation_ids: list[str] = Field(default_factory=list)
    role: str
