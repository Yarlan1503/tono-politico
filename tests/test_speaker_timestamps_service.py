"""Tests de SpeakerTimestampsService (con mocks de pyannote)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tono_politico.speech2text.audio_fetcher.models import AudioVideo
from tono_politico.speech2text.models import PerfilVozActor, SpeakerMatch
from tono_politico.speech2text.speaker_timestamps.service import (
    SpeakerTimestampsService,
    _extraer_turnos,
)


def _audio(tmp_path: Path, video_id: str = "vid1") -> AudioVideo:
    wav = tmp_path / f"{video_id}.wav"
    wav.write_bytes(b"RIFF")
    return AudioVideo(
        video_id=video_id,
        url=f"https://example.com/{video_id}",
        titulo="T",
        fecha=None,
        audio_path=wav,
        duracion=10.0,
    )


def _perfil() -> PerfilVozActor:
    return PerfilVozActor(
        actor="Actor",
        video_id_referencia="ref",
        embedding=[1.0, 0.0],
        modelo_embedding="test",
        duracion_segundos=10.0,
    )


class TestExtraerTurnos:
    def test_from_exclusive(self) -> None:
        seg1 = SimpleNamespace(start=0.0, end=1.5)
        seg2 = SimpleNamespace(start=2.0, end=3.0)
        exclusive = MagicMock()
        exclusive.itertracks.return_value = [
            (seg1, None, "SPEAKER_00"),
            (seg2, None, "SPEAKER_01"),
        ]
        output = SimpleNamespace(exclusive_speaker_diarization=exclusive)
        turnos = _extraer_turnos(output, "v1")
        assert len(turnos) == 2
        assert turnos[0].speaker_id == "SPEAKER_00"
        assert turnos[0].t_end == 1.5


class TestSpeakerTimestampsService:
    def test_procesar_one_sin_perfil_raises(self, tmp_path: Path) -> None:
        svc = SpeakerTimestampsService()
        with pytest.raises(RuntimeError, match="Perfil"):
            svc.procesar_one(_audio(tmp_path))

    def test_procesar_one_filtra_actor(self, tmp_path: Path) -> None:
        svc = SpeakerTimestampsService(actor="Actor")
        svc.set_perfil(_perfil())

        seg_a = SimpleNamespace(start=0.0, end=1.0)
        seg_b = SimpleNamespace(start=1.0, end=2.0)
        exclusive = MagicMock()
        exclusive.itertracks.return_value = [
            (seg_a, None, "SPEAKER_00"),
            (seg_b, None, "SPEAKER_01"),
        ]
        diar = MagicMock()
        diar.labels.return_value = ["SPEAKER_00", "SPEAKER_01"]
        output = SimpleNamespace(
            exclusive_speaker_diarization=exclusive,
            speaker_diarization=diar,
            speaker_embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )

        matches = [
            SpeakerMatch(
                speaker_id="SPEAKER_00",
                distancia=0.1,
                aceptado=True,
                es_ambiguo=False,
            ),
            SpeakerMatch(
                speaker_id="SPEAKER_01",
                distancia=0.9,
                aceptado=False,
                es_ambiguo=False,
            ),
        ]

        with (
            patch.object(svc, "_get_pipeline", return_value="pipe"),
            patch(
                "tono_politico.speech2text.speaker_timestamps.service.run_pyannote_pipeline",
                return_value=output,
            ),
            patch(
                "tono_politico.speech2text.speaker_timestamps.service.identificar_actor",
                return_value=matches,
            ),
        ):
            turnos = svc.procesar_one(_audio(tmp_path))

        assert len(turnos) == 1
        assert turnos[0].speaker_id == "SPEAKER_00"

    def test_procesar_one_rechaza_embeddings_inconsistentes(self, tmp_path: Path) -> None:
        svc = SpeakerTimestampsService()
        svc.set_perfil(_perfil())
        exclusive = MagicMock()
        exclusive.itertracks.return_value = [
            (SimpleNamespace(start=0.0, end=1.0), None, "SPEAKER_00"),
        ]
        diar = MagicMock()
        diar.labels.return_value = ["SPEAKER_00", "SPEAKER_01"]
        output = SimpleNamespace(
            exclusive_speaker_diarization=exclusive,
            speaker_diarization=diar,
            speaker_embeddings=[[1.0, 0.0]],
        )
        with (
            patch.object(svc, "_get_pipeline", return_value="pipe"),
            patch(
                "tono_politico.speech2text.speaker_timestamps.service.run_pyannote_pipeline",
                return_value=output,
            ),
            pytest.raises(ValueError, match="embeddings"),
        ):
            svc.procesar_one(_audio(tmp_path))

    def test_procesar_one_rechaza_turno_invalido(self, tmp_path: Path) -> None:
        svc = SpeakerTimestampsService()
        svc.set_perfil(_perfil())
        exclusive = MagicMock()
        exclusive.itertracks.return_value = [
            (SimpleNamespace(start=2.0, end=1.0), None, "SPEAKER_00"),
        ]
        output = SimpleNamespace(exclusive_speaker_diarization=exclusive)

        with pytest.raises(ValueError, match="t_end"):
            _extraer_turnos(output, "vid1")
