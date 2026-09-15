#!/usr/bin/env python3
"""Strict contracts for every semantic LLM boundary.

The models deliberately allow only the vocabulary the pipeline can validate.
Unknown fields are rejected so a model cannot silently extend the meaning of a
record.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
from semantics.ontology import ClaimKind
from contracts.meeting import Condition, Quantity, TimeExpression


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedFact(StrictModel):
    type: ClaimKind
    topic: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    speaker_refs: list[str] = Field(default_factory=list)
    certainty: Literal["explicit", "tentative"] = "explicit"


class ExtractionResponse(StrictModel):
    facts: list[ExtractedFact] = Field(default_factory=list)
    no_material: bool = False
    coverage_note: str = ""


class SemanticRecordResponse(StrictModel):
    record_id: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    polarity: Literal["positive", "negative"] = "positive"
    modality: Literal["asserted", "tentative", "proposed", "committed", "question"] = "asserted"
    content_kind: Optional[ClaimKind] = None
    speech_act: Optional[Literal["assert", "propose", "ask", "answer", "commit", "accept", "reject", "correct", "decide"]] = None
    conditions: list[Condition] = Field(default_factory=list)
    quantities: list[Quantity] = Field(default_factory=list)
    time_expression: Optional[TimeExpression] = None
    proposed_by: list[str] = Field(default_factory=list)
    assignees: list[str] = Field(default_factory=list)
    confirmation_evidence_ids: list[str] = Field(default_factory=list)
    commitment_strength: Literal["explicit", "implicit", "none"] = "none"
    commitment_actor: Optional[str] = None
    assignment_actor: Optional[str] = None
    assignment_target: Optional[str] = None
    question_status: Literal["resolved", "unresolved", "unclear"] = "unclear"
    answer_evidence_ids: list[str] = Field(default_factory=list)
    answer_record_ids: list[str] = Field(default_factory=list)
    requested_slots: list[str] = Field(default_factory=list)
    answered_slots: list[str] = Field(default_factory=list)


class SemanticBatchResponse(StrictModel):
    records: list[SemanticRecordResponse]


CONTRACTS = {
    "extraction": ExtractionResponse,
    "semantic_records": SemanticBatchResponse,
}


def validate_response(payload, contract):
    model = CONTRACTS[contract] if isinstance(contract, str) else contract
    return model.model_validate(payload).model_dump(mode="json")


def json_schema(contract):
    model = CONTRACTS[contract] if isinstance(contract, str) else contract
    return model.model_json_schema()
