"""Serialización del contrato actor_transcript.v1.

El contrato persistido es actor-only y turn-level: conserva el texto y los
rangos temporales del turno atribuido al actor, sin words, probability,
pausa_antes ni datos verbose de Whisper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ActorTranscript, ActorTranscriptSegment, AsrMetadata

SCHEMA_VERSION = "actor_transcript.v1"
SCOPE_ACTOR_ONLY = "actor_only"


def _segment_to_dict(segment: ActorTranscriptSegment) -> dict[str, Any]:
    return {
        "text": segment.text,
        "t_start": segment.t_start,
        "t_end": segment.t_end,
        "speaker": segment.speaker,
        "source_turn": {
            "t_start": segment.source_turn_start,
            "t_end": segment.source_turn_end,
        },
        "word_count": segment.word_count,
    }


def _segment_from_dict(data: dict[str, Any]) -> ActorTranscriptSegment:
    if "source_turn" in data:
        source_turn = data["source_turn"]
        source_start = source_turn["t_start"]
        source_end = source_turn["t_end"]
    else:
        # contrato plano (smokes / variantes de serialización)
        source_start = data.get("source_turn_start", data["t_start"])
        source_end = data.get("source_turn_end", data["t_end"])
    return ActorTranscriptSegment(
        text=data["text"],
        t_start=data["t_start"],
        t_end=data["t_end"],
        speaker=data["speaker"],
        source_turn_start=source_start,
        source_turn_end=source_end,
        word_count=data["word_count"],
    )


def actor_transcript_to_dict(transcript: ActorTranscript) -> dict[str, Any]:
    """Serializa ``ActorTranscript`` al contrato JSON actor_transcript.v1."""
    data: dict[str, Any] = {
        "schema_version": transcript.schema_version,
        "video_id": transcript.video_id,
        "actor": transcript.actor,
        "scope": transcript.scope,
        "asr": {
            "provider": transcript.asr.provider,
            "model": transcript.asr.model,
            "language": transcript.asr.language,
        },
        "segments": [_segment_to_dict(segment) for segment in transcript.segments],
    }
    if transcript.fecha is not None:
        data["fecha"] = transcript.fecha
    return data


def actor_transcript_to_json(transcript: ActorTranscript) -> str:
    """Serializa ``ActorTranscript`` como JSON compacto UTF-8 friendly."""
    return json.dumps(
        actor_transcript_to_dict(transcript),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def actor_transcript_from_json(json_str: str) -> ActorTranscript:
    """Deserializa un ``ActorTranscript`` desde JSON actor_transcript.v1."""
    data = json.loads(json_str)
    asr = data["asr"]
    return ActorTranscript(
        schema_version=data["schema_version"],
        video_id=data["video_id"],
        actor=data["actor"],
        scope=data["scope"],
        asr=AsrMetadata(
            provider=asr["provider"],
            model=asr["model"],
            language=asr["language"],
        ),
        segments=[_segment_from_dict(segment) for segment in data.get("segments", [])],
        fecha=data.get("fecha"),
    )


def guardar_actor_transcript(transcript: ActorTranscript, path: Path | str) -> Path:
    """Persiste un ``ActorTranscript`` en ``path`` y devuelve la ruta."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(actor_transcript_to_json(transcript), encoding="utf-8")
    return path


def cargar_actor_transcript(path: Path | str) -> ActorTranscript:
    """Carga un ``ActorTranscript`` desde ``path``."""
    path = Path(path)
    return actor_transcript_from_json(path.read_text(encoding="utf-8"))
