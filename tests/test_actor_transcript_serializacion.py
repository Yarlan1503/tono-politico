"""Tests del contrato persistido actor_transcript.v1."""

from __future__ import annotations

import json
from pathlib import Path

from tono_politico.execution.artifacts import (
    actor_transcript_from_json,
    actor_transcript_to_dict,
    actor_transcript_to_json,
    cargar_actor_transcript,
    guardar_actor_transcript,
)
from tono_politico.speech2text.models import (
    ActorTranscript,
    ActorTranscriptSegment,
    AsrMetadata,
)


def _actor_transcript() -> ActorTranscript:
    return ActorTranscript(
        schema_version="actor_transcript.v1",
        video_id="su9nURIj9XQ",
        actor="Lilly Téllez",
        scope="actor_only",
        asr=AsrMetadata(
            provider="whisper",
            model="large-v3-turbo",
            language="es",
        ),
        segments=[
            ActorTranscriptSegment(
                text="Gracias. Acabamos de escuchar un recuento.",
                t_start=0.0,
                t_end=6.58,
                speaker="SPEAKER_01",
                source_turn_start=0.0,
                source_turn_end=7.0,
                word_count=7,
            ),
        ],
    )


def test_actor_transcript_to_dict_expone_contrato_actor_only():
    data = actor_transcript_to_dict(_actor_transcript())

    assert data["schema_version"] == "actor_transcript.v1"
    assert data["video_id"] == "su9nURIj9XQ"
    assert data["actor"] == "Lilly Téllez"
    assert data["scope"] == "actor_only"
    assert data["asr"] == {
        "provider": "whisper",
        "model": "large-v3-turbo",
        "language": "es",
    }
    assert data["segments"] == [
        {
            "text": "Gracias. Acabamos de escuchar un recuento.",
            "t_start": 0.0,
            "t_end": 6.58,
            "speaker": "SPEAKER_01",
            "source_turn": {
                "t_start": 0.0,
                "t_end": 7.0,
            },
            "word_count": 7,
        }
    ]


def test_actor_transcript_json_no_persiste_campos_descartados():
    payload = actor_transcript_to_json(_actor_transcript())
    data = json.loads(payload)

    assert "words" not in payload
    assert "probability" not in payload
    assert "pausa_antes" not in payload
    assert "verbose" not in payload
    assert "captions" not in payload
    assert "segments" in data


def test_actor_transcript_round_trip_preserva_datos_de_turno():
    original = _actor_transcript()

    recovered = actor_transcript_from_json(actor_transcript_to_json(original))

    assert recovered == original
    assert recovered.segments[0].speaker == "SPEAKER_01"
    assert recovered.segments[0].source_turn_end == 7.0


def test_guardar_y_cargar_actor_transcript(tmp_path: Path):
    original = _actor_transcript()
    path = tmp_path / "actor_transcript.json"

    saved_path = guardar_actor_transcript(original, path)
    recovered = cargar_actor_transcript(saved_path)

    assert saved_path == path
    assert path.exists()
    assert recovered.video_id == "su9nURIj9XQ"
    assert recovered.asr.model == "large-v3-turbo"


def test_cargar_rechaza_schema_version_desconocido(tmp_path: Path):
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps({"schema_version": "actor_transcript.v0"}),
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(ValueError, match="schema_version"):
        cargar_actor_transcript(path)


def test_cargar_rechaza_segmento_con_rango_invertido(tmp_path: Path):
    payload = {
        "schema_version": "actor_transcript.v1",
        "video_id": "video",
        "actor": "Actor",
        "scope": "actor_only",
        "asr": {"provider": "whisper", "model": "tiny", "language": "es"},
        "segments": [
            {
                "text": "Texto",
                "t_start": 2.0,
                "t_end": 1.0,
                "speaker": "SPEAKER_00",
                "source_turn": {"t_start": 1.0, "t_end": 2.0},
                "word_count": 1,
            }
        ],
    }
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="t_end"):
        cargar_actor_transcript(path)


def test_cargar_rechaza_scope_no_actor_only(tmp_path: Path):
    payload = json.loads(actor_transcript_to_json(_actor_transcript()))
    payload["scope"] = "full_video"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="scope"):
        cargar_actor_transcript(path)
