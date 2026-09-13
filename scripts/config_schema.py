#!/usr/bin/env python3
"""Single typed configuration boundary for every runtime entry point."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    audio_track: int = Field(0, ge=0)
    poll_seconds: float = Field(10, gt=0)
    stable_seconds: float = Field(30, ge=0)
    asr_device: str = "auto"
    diarization_device: str = "auto"
    diarization_model: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
    diarization_model_revision: str = "main"
    diarization_batch_size: int = Field(8, ge=1)
    diarization_min_speakers: int = Field(1, ge=1, le=8)
    diarization_max_speakers: int = Field(8, ge=1, le=8)
    ultra_model: str = "mago-ai/ultra_diar_streaming_sortformer_8spk_v1"
    ultra_model_revision: str = "main"
    ultra_device: str = "auto"
    boundary_tolerance_ms: int = Field(300, ge=0)
    redimnet_repository: str = "PalabraAI/redimnet2"
    redimnet_revision: str = "c5bbe0b76e37df698c403f8844e41304ceab6307"
    redimnet_device: str = "auto"
    redimnet_anchor_min_seconds: float = 3.0
    redimnet_anchor_target_seconds: float = 30.0
    redimnet_known_threshold: float = 0.55
    redimnet_known_margin: float = 0.08
    redimnet_phrase_min_seconds: float = 1.8
    redimnet_phrase_max_seconds: float = 24.0
    redimnet_phrase_threshold: float = 0.78
    redimnet_phrase_margin: float = 0.2
    redimnet_corroborated_phrase_threshold: float = 0.58
    redimnet_corroborated_phrase_margin: float = 0.16
    redimnet_corroborated_phrase_coverage: float = 0.72
    redimnet_second_pass_min_seconds: float = 1.8
    redimnet_second_pass_context_seconds: float = 0.35
    redimnet_second_pass_track_context_seconds: float = 1.5
    redimnet_second_pass_threshold: float = 0.74
    redimnet_second_pass_margin: float = 0.18
    voice_identity_boundary_gap_seconds: float = 0.25
    short_turn_seconds: float = 1.5
    word_boundary_context_seconds: float = 0.4
    gigaam_model: str = "v3_e2e_rnnt"
    language: str = "ru"
    vad_threshold: float = Field(0.42, ge=0, le=1)
    vad_min_speech_ms: int = Field(180, ge=0)
    vad_min_silence_ms: int = Field(320, ge=0)
    vad_speech_pad_ms: int = Field(220, ge=0)
    asr_chunk_seconds: float = Field(22.0, gt=1)
    asr_overlap_seconds: float = Field(0.45, ge=0)
    notify: bool = False
    dashboard_port: int = Field(8765, ge=1, le=65535)
    open_dashboard_on_job: bool = False
    processing_enabled: bool = True
    speaker_match_min_overlap: float = 0.5
    speaker_match_margin: float = 0.15
    voice_match_threshold: float = 0.62
    voice_match_margin: float = 0.08
    voice_embedding_device: str = "auto"
    max_profile_sample_seconds: float = 600
    voice_meeting_samples_per_speaker: int = 8
    voice_identity_min_phrase_seconds: float = 2.0
    voice_identity_max_phrase_seconds: float = 10.0
    voice_identity_phrase_gap_seconds: float = 0.65
    voice_identity_max_window_seconds: float = 13.0
    voice_identity_merge_gap_seconds: float = 1.8
    voice_identity_threshold: float = 0.54
    voice_identity_profile_separation: float = 0.0
    voice_identity_margin: float = 0.12
    voice_identity_strong_threshold: float = 0.72
    voice_identity_strong_margin: float = 0.08
    voice_identity_short_window_seconds: float = 4.0
    voice_identity_short_threshold: float = 0.72
    voice_identity_short_margin: float = 0.14
    voice_cluster_min_evidence_seconds: float = 10.0
    voice_cluster_dominance: float = 0.82
    voice_cluster_runner_max: float = 0.15
    speaker_island_max_seconds: float = 2.2
    speaker_phrase_dominance: float = 0.55
    speaker_phrase_low_confidence_dominance: float = 0.52
    clause_coherence_gap_seconds: float = 2.5
    clause_coherence_short_seconds: float = 3.0
    clause_coherence_acoustic_gap_seconds: float = 0.35
    clause_unanchored_prefix_max_seconds: float = 1.5
    clause_unanchored_prefix_gap_seconds: float = 2.0
    voice_identity_context_max_seconds: float = 2.0
    voice_identity_context_gap_seconds: float = 0.3
    voice_identity_same_context_gap_seconds: float = 4.0
    voice_identity_punctuation_gap_seconds: float = 2.0
    utterance_gap_seconds: float = 1.2
    utterance_continuation_gap_seconds: float = 3.0
    summary_enabled: bool = True
    summary_project_name: str = "Project"
    ollama_url: str = "http://127.0.0.1:11434"
    summary_extractor_model: str = "qwen3.5:9b-q4_K_M"
    summary_arbitrator_model: str = "qwen3.5:9b-q4_K_M"
    summary_high_risk_verifier_model: str = "ministral-3:14b-instruct-2512-q4_K_M"
    summary_critical_secondary_verifier_model: str = "gemma3:12b"
    summary_writer_model: str = "qwen3.5:9b-q4_K_M"
    summary_auditor_model: str = "qwen3.5:9b-q4_K_M"
    summary_public_auditor_model: str = "qwen3.8:27b-q4_K_M"
    summary_segment_target_seconds: float = 300
    summary_segment_min_seconds: float = 120
    summary_segment_max_seconds: float = 480
    summary_halo_seconds: float = 35
    summary_extract_attempts: int = Field(3, ge=1)
    summary_material_char_threshold: int = Field(500, ge=0)
    summary_validation_batch_size: int = Field(18, ge=1)
    summary_arbitration_batch_size: int = Field(12, ge=1)
    summary_min_coverage: float = Field(0.995, ge=0, le=1)
    summary_writer_context: int = Field(16384, ge=2048)
    summary_auditor_context: int = Field(16384, ge=2048)
    summary_min_material_coverage: float = Field(0.8, ge=0, le=1)
    summary_min_publication_coverage: float = Field(0.99, ge=0, le=1)
    summary_resolution_batch_size: int = Field(7, ge=1)
    summary_auditor_failure_policy: Literal["risk_based", "fail_closed"] = "risk_based"
    summary_audio_repair_enabled: bool = True
    summary_repair_padding_before_seconds: float = Field(2.0, ge=0, le=10)
    summary_repair_padding_after_seconds: float = Field(4.0, ge=0, le=15)
    summary_repair_max_windows: int = Field(24, ge=0, le=100)
    summary_require_immutable_provenance: bool = True
    summary_public_fact_limit: int = Field(32, ge=8, le=80)
    summary_navigation_max_chapters: int = Field(12, ge=4, le=16)
    domain_vocabulary: dict[str, str] = Field(default_factory=dict)

    def model_post_init(self, __context) -> None:
        if self.diarization_min_speakers > self.diarization_max_speakers:
            raise ValueError("diarization_min_speakers cannot exceed diarization_max_speakers")
        if not self.summary_segment_min_seconds <= self.summary_segment_target_seconds <= self.summary_segment_max_seconds:
            raise ValueError("summary segment bounds must satisfy min <= target <= max")
        if self.asr_overlap_seconds >= self.asr_chunk_seconds:
            raise ValueError("asr_overlap_seconds must be smaller than asr_chunk_seconds")
        if self.redimnet_anchor_min_seconds > self.redimnet_anchor_target_seconds:
            raise ValueError("redimnet anchor bounds must satisfy min <= target")
        if self.redimnet_phrase_min_seconds > self.redimnet_phrase_max_seconds:
            raise ValueError("redimnet phrase bounds must satisfy min <= max")
        if self.voice_identity_min_phrase_seconds > self.voice_identity_max_phrase_seconds:
            raise ValueError("voice identity phrase bounds must satisfy min <= max")
        if self.voice_identity_strong_threshold < self.voice_identity_threshold:
            raise ValueError("strong voice threshold must not be lower than the normal threshold")


def load_config(path: Path, resolved_path: Path | None = None) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    resolved = PipelineConfig.model_validate(raw).model_dump(mode="json")
    if resolved_path is not None:
        resolved_path = Path(resolved_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = resolved_path.with_suffix(resolved_path.suffix + ".tmp")
        temporary.write_text(json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(resolved_path)
    return resolved
