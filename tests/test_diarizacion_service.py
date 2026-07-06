"""Tests para DiarizacionService: integración del Componente 1.5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from tono_politico.diarizacion.models import TurnoOrador
from tono_politico.diarizacion.service import DiarizacionService
from tono_politico.models import SegmentoRaw, VideoTranscript, WordTimestamp

# ──────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────


@dataclass
class FakeSegment:
    start: float
    end: float


@dataclass
class FakeExclusiveDiarization:
    tracks: list[tuple[FakeSegment, str, str]]

    def itertracks(self, yield_label=True):
        for seg, track, label in self.tracks:
            if yield_label:
                yield seg, track, label
            else:
                yield seg, track


@dataclass
class FakeDiarizationOutput:
    _exclusive: FakeExclusiveDiarization

    @property
    def exclusive_speaker_diarization(self) -> FakeExclusiveDiarization:
        return self._exclusive


class FakePipeline:
    """Pipeline que simula diarización para múltiples audios."""

    def __init__(self, tracks_by_audio: dict[str, list[tuple[float, float, str]]]):
        self._tracks = tracks_by_audio

    def __call__(self, audio_path, **kwargs) -> FakeDiarizationOutput:
        key = str(audio_path)
        raw_tracks = self._tracks.get(key, [])
        tracks = [
            (FakeSegment(s, e), f"track_{i}", label)
            for i, (s, e, label) in enumerate(raw_tracks)
        ]
        return FakeDiarizationOutput(FakeExclusiveDiarization(tracks))


class FakeEmbeddingPipeline:
    """Embedding pipeline que devuelve embeddings por audio path."""

    def __init__(self, emb_by_audio: dict[str, list[float]]):
        self._embs = emb_by_audio

    def __call__(self, audio_path) -> FakeEmbeddingPipeline:
        self._current = self._embs[str(audio_path)]
        return self

    @property
    def shape(self):
        return (1, len(self._current))

    def __getitem__(self, idx):
        return self._current if idx == 0 else []

    def tolist(self):
        return [self._current]


class FakeAudioHelper:
    def __init__(self, durations: dict[str, float]):
        self._durations = durations

    def get_duration(self, file_path) -> float:
        return self._durations.get(str(file_path), 30.0)


class FakeEmbeddingExtractor:
    """Simula la extracción de embedding promedio por speaker.

    Devuelve un embedding controlable por speaker_id.
    """

    def __init__(self, emb_by_speaker: dict[str, list[float]]):
        self._embs = emb_by_speaker

    def __call__(
        self, audio_path, turnos: list[TurnoOrador]
    ) -> dict[str, list[float]]:
        return {
            speaker_id: self._embs[speaker_id]
            for speaker_id in {t.speaker_id for t in turnos}
            if speaker_id in self._embs
        }


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────


def _seg(texto: str, t_start: float, t_end: float) -> SegmentoRaw:
    return SegmentoRaw(
        texto=texto,
        t_start=t_start,
        t_end=t_end,
        pausa_antes=0.0,
        words=[WordTimestamp(word="x", start=t_start, end=t_end)],
    )


def _transcript(segments: list[SegmentoRaw], video_id: str = "v1") -> VideoTranscript:
    return VideoTranscript(
        video_id=video_id,
        url=f"https://youtube.com/watch?v={video_id}",
        titulo="Test",
        fecha="20260101",
        raw_segments=segments,
    )


# ──────────────────────────────────────────────────────────
# Tests: constructor y defaults
# ──────────────────────────────────────────────────────────


class TestConstructor:
    """DiarizacionService defaults y encapsulación de config."""

    def test_defaults(self):
        """Los defaults del constructor son correctos."""
        svc = DiarizacionService()
        assert svc.actor == "Lilly Téllez"
        assert svc.video_ref_id == "su9nURIj9XQ"
        assert svc.umbral_match == pytest.approx(0.5)
        assert svc.umbral_ambiguo == pytest.approx(0.7)

    def test_config_personalizada(self):
        """El constructor acepta config personalizada."""
        svc = DiarizacionService(
            actor="X",
            video_ref_id="abc",
            umbral_match=0.3,
            umbral_ambiguo=0.6,
        )
        assert svc.actor == "X"
        assert svc.video_ref_id == "abc"
        assert svc.umbral_match == pytest.approx(0.3)

    def test_data_dir_default(self):
        svc = DiarizacionService()
        assert svc.data_dir == Path("data")


# ──────────────────────────────────────────────────────────
# Tests: procesar — un video, un speaker aceptado
# ──────────────────────────────────────────────────────────


class TestProcesarUnVideo:
    """procesar() con un video donde el actor habla."""

    def test_un_video_actor_identificado(self):
        """Un video con un speaker que matchea el perfil → segmentos filtrados."""
        svc = DiarizacionService()
        playlist = "test"

        transcript = _transcript([
            _seg("Hola", 0.0, 2.0),       # midpoint 1.0 → turno 0-3
            _seg("Mundo", 4.0, 6.0),       # midpoint 5.0 → turno 4-7
        ])

        fake_pipeline = FakePipeline({
            str(svc._audio_path("v1", playlist)): [
                (0.0, 3.0, "SPEAKER_00"),
                (4.0, 7.0, "SPEAKER_00"),
            ],
        })
        fake_ref_emb = [1.0, 0.0, 0.0]

        with (
            patch.object(svc, "_get_pipeline", return_value=fake_pipeline),
            patch.object(svc, "_get_embedding_pipeline", return_value=FakeEmbeddingPipeline({
                str(svc._ref_audio_path()): fake_ref_emb,
            })),
            patch.object(svc, "_get_audio_helper", return_value=FakeAudioHelper({
                str(svc._ref_audio_path()): 30.0,
            })),
            patch.object(svc, "_get_embedding_extractor", return_value=FakeEmbeddingExtractor({
                "SPEAKER_00": [0.99, 0.01, 0.0],
            })),
        ):
            resultado = svc.procesar([transcript], nombre_playlist=playlist)

        assert len(resultado) == 1
        assert resultado[0].video_id == "v1"
        assert len(resultado[0].raw_segments) == 2

    def test_un_video_actor_no_identificado(self):
        """Ningún speaker matchea → transcript con 0 segmentos."""
        svc = DiarizacionService()
        playlist = "test"

        transcript = _transcript([_seg("Hola", 0.0, 2.0)])

        fake_pipeline = FakePipeline({
            str(svc._audio_path("v1", playlist)): [(0.0, 3.0, "SPEAKER_00")],
        })

        with (
            patch.object(svc, "_get_pipeline", return_value=fake_pipeline),
            patch.object(svc, "_get_embedding_pipeline", return_value=FakeEmbeddingPipeline({
                str(svc._ref_audio_path()): [1.0, 0.0, 0.0],
            })),
            patch.object(svc, "_get_audio_helper", return_value=FakeAudioHelper({
                str(svc._ref_audio_path()): 30.0,
            })),
            patch.object(svc, "_get_embedding_extractor", return_value=FakeEmbeddingExtractor({
                "SPEAKER_00": [0.0, 1.0, 0.0],  # ortogonal → rechazado
            })),
        ):
            resultado = svc.procesar([transcript], nombre_playlist=playlist)

        assert len(resultado) == 1
        assert len(resultado[0].raw_segments) == 0


# ──────────────────────────────────────────────────────────
# Tests: procesar — múltiples videos
# ──────────────────────────────────────────────────────────


class TestProcesarMultiplesVideos:
    """procesar() itera videos independientemente."""

    def test_dos_videos_actor_en_ambos(self):
        """Actor presente en ambos videos."""
        svc = DiarizacionService()
        playlist = "test"

        t1 = _transcript([_seg("A", 0.0, 1.0)], video_id="v1")
        t2 = _transcript([_seg("B", 0.0, 1.0)], video_id="v2")

        fake_pipeline = FakePipeline({
            str(svc._audio_path("v1", playlist)): [(0.0, 2.0, "SPEAKER_00")],
            str(svc._audio_path("v2", playlist)): [(0.0, 2.0, "SPEAKER_01")],
        })

        with (
            patch.object(svc, "_get_pipeline", return_value=fake_pipeline),
            patch.object(svc, "_get_embedding_pipeline", return_value=FakeEmbeddingPipeline({
                str(svc._ref_audio_path()): [1.0, 0.0, 0.0],
            })),
            patch.object(svc, "_get_audio_helper", return_value=FakeAudioHelper({
                str(svc._ref_audio_path()): 30.0,
            })),
            patch.object(svc, "_get_embedding_extractor", return_value=FakeEmbeddingExtractor({
                "SPEAKER_00": [0.99, 0.01, 0.0],
                "SPEAKER_01": [0.98, 0.02, 0.0],
            })),
        ):
            resultado = svc.procesar([t1, t2], nombre_playlist=playlist)

        assert len(resultado) == 2
        assert len(resultado[0].raw_segments) == 1
        assert len(resultado[1].raw_segments) == 1

    def test_un_video_con_actor_otro_sin(self):
        """v1 tiene al actor, v2 no."""
        svc = DiarizacionService()
        playlist = "test"

        t1 = _transcript([_seg("Actor", 0.0, 1.0)], video_id="v1")
        t2 = _transcript([_seg("Otro", 0.0, 1.0)], video_id="v2")

        fake_pipeline = FakePipeline({
            str(svc._audio_path("v1", playlist)): [(0.0, 2.0, "SPEAKER_00")],
            str(svc._audio_path("v2", playlist)): [(0.0, 2.0, "SPEAKER_00")],
        })

        with (
            patch.object(svc, "_get_pipeline", return_value=fake_pipeline),
            patch.object(svc, "_get_embedding_pipeline", return_value=FakeEmbeddingPipeline({
                str(svc._ref_audio_path()): [1.0, 0.0, 0.0],
            })),
            patch.object(svc, "_get_audio_helper", return_value=FakeAudioHelper({
                str(svc._ref_audio_path()): 30.0,
            })),
            patch.object(svc, "_get_embedding_extractor", return_value=FakeEmbeddingExtractor({
                # v1 matchea, v2 no
                "SPEAKER_00": [0.0, 1.0, 0.0],
            })),
        ):
            resultado = svc.procesar([t1, t2], nombre_playlist=playlist)

        assert len(resultado[0].raw_segments) == 0
        assert len(resultado[1].raw_segments) == 0


# ──────────────────────────────────────────────────────────
# Tests: edge cases
# ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """procesar() con entradas vacías."""

    def test_lista_vacia(self):
        """Lista vacía de transcripts → lista vacía."""
        svc = DiarizacionService()

        with (
            patch.object(svc, "_get_pipeline", return_value=FakePipeline({})),
            patch.object(svc, "_get_embedding_pipeline", return_value=FakeEmbeddingPipeline({
                str(svc._ref_audio_path()): [1.0, 0.0, 0.0],
            })),
            patch.object(svc, "_get_audio_helper", return_value=FakeAudioHelper({
                str(svc._ref_audio_path()): 30.0,
            })),
            patch.object(svc, "_get_embedding_extractor", return_value=FakeEmbeddingExtractor({})),
        ):
            resultado = svc.procesar([], nombre_playlist="test")

        assert resultado == []

    def test_video_sin_segments(self):
        """Video con raw_segments=[] → transcript con 0 segmentos."""
        svc = DiarizacionService()
        playlist = "test"

        transcript = _transcript([], video_id="v1")

        fake_pipeline = FakePipeline({
            str(svc._audio_path("v1", playlist)): [(0.0, 2.0, "SPEAKER_00")],
        })

        with (
            patch.object(svc, "_get_pipeline", return_value=fake_pipeline),
            patch.object(svc, "_get_embedding_pipeline", return_value=FakeEmbeddingPipeline({
                str(svc._ref_audio_path()): [1.0, 0.0, 0.0],
            })),
            patch.object(svc, "_get_audio_helper", return_value=FakeAudioHelper({
                str(svc._ref_audio_path()): 30.0,
            })),
            patch.object(svc, "_get_embedding_extractor", return_value=FakeEmbeddingExtractor({
                "SPEAKER_00": [0.99, 0.01, 0.0],
            })),
        ):
            resultado = svc.procesar([transcript], nombre_playlist=playlist)

        assert len(resultado) == 1
        assert len(resultado[0].raw_segments) == 0
