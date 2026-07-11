"""Tests de recorte y transcripción de turnos actor-only con Whisper mockeable."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

from tono_politico.speech2text.models import TurnoOrador


def _sut():
    """Carga el módulo bajo prueba solo al ejecutar cada test.

    Esto permite que pytest colecte todos los tests RED aunque el módulo aún
    no exista: la falla esperada inicial es ModuleNotFoundError.
    """
    return import_module("tono_politico.speech2text.transcribe_speech.actor_clip")


def _clip(text: str, t_start: float, t_end: float):
    return _sut().ClipTranscriptSegment(text=text, t_start=t_start, t_end=t_end)


class FakeClipTranscriber:
    def __init__(self, responses: list[list[object]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def transcribir_clip(
        self,
        audio_path: Path,
        *,
        t_start: float,
        t_end: float,
        modelo: str,
        idioma: str,
    ) -> list[object]:
        self.calls.append(
            {
                "audio_path": audio_path,
                "t_start": t_start,
                "t_end": t_end,
                "modelo": modelo,
                "idioma": idioma,
            }
        )
        return self.responses.pop(0)


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "video.wav"
    path.write_bytes(b"fake-audio")
    return path


def test_transcribe_unicamente_los_rangos_de_turnos_actor(tmp_path: Path):
    audio_path = _audio(tmp_path)
    transcriptor = FakeClipTranscriber(
        responses=[
            [_clip("Primer turno del actor.", 0.0, 4.8)],
            [_clip("Segundo turno del actor.", 0.0, 3.1)],
        ]
    )

    result = _sut().transcribir_turnos_actor(
        audio_path,
        [
            TurnoOrador("video-1", "SPEAKER_01", 10.0, 15.0),
            TurnoOrador("video-1", "SPEAKER_01", 30.0, 35.0),
        ],
        video_id="video-1",
        actor="Lilly Téllez",
        transcriptor=transcriptor,
        modelo="large-v3-turbo",
        idioma="es",
    )

    assert [call["t_start"] for call in transcriptor.calls] == [10.0, 30.0]
    assert [call["t_end"] for call in transcriptor.calls] == [15.0, 35.0]
    assert [call["modelo"] for call in transcriptor.calls] == ["large-v3-turbo"] * 2
    assert [call["idioma"] for call in transcriptor.calls] == ["es"] * 2
    assert result.video_id == "video-1"
    assert result.actor == "Lilly Téllez"
    assert result.asr.model == "large-v3-turbo"
    assert len(result.segments) == 2


def test_convierte_timestamps_relativos_del_clip_a_timeline_absoluto(tmp_path: Path):
    audio_path = _audio(tmp_path)
    transcriptor = FakeClipTranscriber(
        responses=[[_clip("Estamos defendiendo la salud pública.", 0.5, 3.0)]]
    )

    result = _sut().transcribir_turnos_actor(
        audio_path,
        [TurnoOrador("video-1", "SPEAKER_01", 100.0, 110.0)],
        video_id="video-1",
        actor="Lilly Téllez",
        transcriptor=transcriptor,
    )

    segment = result.segments[0]
    assert segment.t_start == pytest.approx(100.5)
    assert segment.t_end == pytest.approx(103.0)


def test_conserva_limites_originales_de_pyannote_como_source_turn(tmp_path: Path):
    audio_path = _audio(tmp_path)
    transcriptor = FakeClipTranscriber(responses=[[_clip("Texto del actor.", 0.4, 2.0)]])

    result = _sut().transcribir_turnos_actor(
        audio_path,
        [TurnoOrador("video-1", "SPEAKER_07", 42.0, 50.0)],
        video_id="video-1",
        actor="Lilly Téllez",
        transcriptor=transcriptor,
    )

    segment = result.segments[0]
    assert segment.speaker == "SPEAKER_07"
    assert segment.source_turn_start == pytest.approx(42.0)
    assert segment.source_turn_end == pytest.approx(50.0)


def test_padding_solo_modifica_clip_y_no_contamina_source_turn(tmp_path: Path):
    audio_path = _audio(tmp_path)
    transcriptor = FakeClipTranscriber(
        responses=[[_clip("Texto capturado con padding.", 0.2, 11.5)]]
    )

    result = _sut().transcribir_turnos_actor(
        audio_path,
        [TurnoOrador("video-1", "SPEAKER_01", 100.0, 110.0)],
        video_id="video-1",
        actor="Lilly Téllez",
        transcriptor=transcriptor,
        padding_segundos=1.0,
    )

    assert transcriptor.calls[0]["t_start"] == pytest.approx(99.0)
    assert transcriptor.calls[0]["t_end"] == pytest.approx(111.0)

    segment = result.segments[0]
    assert segment.t_start == pytest.approx(100.0)
    assert segment.t_end == pytest.approx(110.0)
    assert segment.source_turn_start == pytest.approx(100.0)
    assert segment.source_turn_end == pytest.approx(110.0)


def test_padding_respeta_inicio_cero_y_duracion_audio(tmp_path: Path):
    audio_path = _audio(tmp_path)
    transcriptor = FakeClipTranscriber(responses=[[_clip("Texto cercano a bordes.", 0.0, 10.0)]])

    _sut().transcribir_turnos_actor(
        audio_path,
        [TurnoOrador("video-1", "SPEAKER_01", 0.2, 9.5)],
        video_id="video-1",
        actor="Lilly Téllez",
        transcriptor=transcriptor,
        padding_segundos=1.0,
        duracion_audio=10.0,
    )

    assert transcriptor.calls[0]["t_start"] == pytest.approx(0.0)
    assert transcriptor.calls[0]["t_end"] == pytest.approx(10.0)


def test_omite_turnos_con_transcripcion_vacia(tmp_path: Path):
    audio_path = _audio(tmp_path)
    transcriptor = FakeClipTranscriber(responses=[[_clip("   ", 0.0, 1.0)]])

    result = _sut().transcribir_turnos_actor(
        audio_path,
        [TurnoOrador("video-1", "SPEAKER_01", 10.0, 15.0)],
        video_id="video-1",
        actor="Lilly Téllez",
        transcriptor=transcriptor,
    )

    assert result.segments == []


def test_rechaza_turnos_de_otro_video(tmp_path: Path):
    audio_path = _audio(tmp_path)
    transcriptor = FakeClipTranscriber(responses=[])

    with pytest.raises(ValueError, match="video_id"):
        _sut().transcribir_turnos_actor(
            audio_path,
            [TurnoOrador("otro-video", "SPEAKER_01", 10.0, 15.0)],
            video_id="video-1",
            actor="Lilly Téllez",
            transcriptor=transcriptor,
        )


def test_rechaza_turnos_con_rango_temporal_invalido(tmp_path: Path):
    audio_path = _audio(tmp_path)
    transcriptor = FakeClipTranscriber(responses=[])

    with pytest.raises(ValueError, match="t_end"):
        _sut().transcribir_turnos_actor(
            audio_path,
            [TurnoOrador("video-1", "SPEAKER_01", 15.0, 10.0)],
            video_id="video-1",
            actor="Lilly Téllez",
            transcriptor=transcriptor,
        )


def test_rechaza_padding_negativo(tmp_path: Path):
    audio_path = _audio(tmp_path)
    transcriptor = FakeClipTranscriber(responses=[])

    with pytest.raises(ValueError, match="padding"):
        _sut().transcribir_turnos_actor(
            audio_path,
            [TurnoOrador("video-1", "SPEAKER_01", 10.0, 15.0)],
            video_id="video-1",
            actor="Lilly Téllez",
            transcriptor=transcriptor,
            padding_segundos=-0.1,
        )


def test_falla_claramente_si_audio_no_existe(tmp_path: Path):
    transcriptor = FakeClipTranscriber(responses=[])

    with pytest.raises(FileNotFoundError, match="Audio"):
        _sut().transcribir_turnos_actor(
            tmp_path / "missing.wav",
            [TurnoOrador("video-1", "SPEAKER_01", 10.0, 15.0)],
            video_id="video-1",
            actor="Lilly Téllez",
            transcriptor=transcriptor,
        )
