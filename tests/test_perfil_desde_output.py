"""Tests para construir_perfil_desde_output — perfil desde speaker_embeddings público."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from tono_politico.diarizacion.perfil_voz import construir_perfil_desde_output


@dataclass
class FakeSegment:
    start: float
    end: float


class FakeDiarization:
    def __init__(self, tracks: list[tuple[FakeSegment, str, str]]):
        self._tracks = tracks

    def itertracks(self, yield_label=True):
        for seg, track, label in self._tracks:
            if yield_label:
                yield seg, track, label
            else:
                yield seg, track

    def labels(self):
        return sorted({label for _, _, label in self._tracks})


class FakeExclusiveDiarization(FakeDiarization):
    pass


@dataclass
class FakeOutput:
    _exclusive: FakeExclusiveDiarization
    _speaker_dia: FakeDiarization
    speaker_embeddings: np.ndarray | None

    @property
    def exclusive_speaker_diarization(self):
        return self._exclusive

    @property
    def speaker_diarization(self):
        return self._speaker_dia


class TestConstruirPerfilDesdeOutput:
    def test_un_speaker_usa_ese_embedding(self):
        output = FakeOutput(
            _exclusive=FakeExclusiveDiarization(
                [
                    (FakeSegment(0, 10), "t0", "SPEAKER_00"),
                ]
            ),
            _speaker_dia=FakeDiarization(
                [
                    (FakeSegment(0, 10), "t0", "SPEAKER_00"),
                ]
            ),
            speaker_embeddings=np.array([[1.0, 0.0, 0.0]]),
        )

        perfil = construir_perfil_desde_output(
            output, actor="Lilly", video_ref_id="ref", pipeline_name="community-1"
        )

        assert perfil.actor == "Lilly"
        assert perfil.video_id_referencia == "ref"
        assert perfil.embedding == [1.0, 0.0, 0.0]
        assert perfil.modelo_embedding == "speaker_embeddings:community-1"
        assert perfil.duracion_segundos == pytest.approx(10.0)

    def test_dos_speakers_elige_mayor_duracion(self):
        output = FakeOutput(
            _exclusive=FakeExclusiveDiarization(
                [
                    (FakeSegment(0, 5), "t0", "SPEAKER_00"),
                    (FakeSegment(5, 20), "t1", "SPEAKER_01"),
                ]
            ),
            _speaker_dia=FakeDiarization(
                [
                    (FakeSegment(0, 5), "t0", "SPEAKER_00"),
                    (FakeSegment(5, 20), "t1", "SPEAKER_01"),
                ]
            ),
            speaker_embeddings=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        )

        perfil = construir_perfil_desde_output(
            output, actor="Lilly", video_ref_id="ref", pipeline_name="community-1"
        )

        assert perfil.embedding == [0.0, 1.0, 0.0]

    def test_sin_embeddings_lanza_error_accionable(self):
        output = FakeOutput(
            _exclusive=FakeExclusiveDiarization(
                [
                    (FakeSegment(0, 10), "t0", "SPEAKER_00"),
                ]
            ),
            _speaker_dia=FakeDiarization(
                [
                    (FakeSegment(0, 10), "t0", "SPEAKER_00"),
                ]
            ),
            speaker_embeddings=None,
        )

        with pytest.raises(ValueError, match="embeddings"):
            construir_perfil_desde_output(
                output, actor="Lilly", video_ref_id="ref", pipeline_name="community-1"
            )
