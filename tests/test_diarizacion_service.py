"""Tests para DiarizacionService: integración del Componente 1.5."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

import tono_politico.diarizacion.service as service_module
from tono_politico.diarizacion.adapter import LoadedPyannotePipeline
from tono_politico.diarizacion.service import DiarizacionService
from tono_politico.models import SegmentoRaw, VideoTranscript, WordTimestamp

FAKE_CREDENTIAL = "HF_TEST_VALUE"

# ──────────────────────────────────────────────────────────
# Fakes que simulan el output completo de pyannote pipeline
# ──────────────────────────────────────────────────────────


@dataclass
class FakeSegment:
    start: float
    end: float


class FakeDiarization:
    """Simula output.speaker_diarization con itertracks y labels."""

    def __init__(self, tracks: list[tuple[FakeSegment, str, str]]):
        self._tracks = tracks

    def itertracks(self, yield_label=True):
        for seg, track, label in self._tracks:
            if yield_label:
                yield seg, track, label
            else:
                yield seg, track

    def labels(self):
        return sorted(set(label for _, _, label in self._tracks))


class FakeExclusiveDiarization(FakeDiarization):
    """Simula output.exclusive_speaker_diarization."""
    pass


@dataclass
class FakePipelineOutput:
    """Simula el objeto devuelto por pipeline(audio_path)."""
    _exclusive: FakeExclusiveDiarization
    _speaker_dia: FakeDiarization
    speaker_embeddings: np.ndarray | None

    @property
    def exclusive_speaker_diarization(self) -> FakeExclusiveDiarization:
        return self._exclusive

    @property
    def speaker_diarization(self) -> FakeDiarization:
        return self._speaker_dia


class FakeEmbCallable:
    """Simula pipeline._inferences['_embedding'] — callable que devuelve embedding."""

    def __init__(self, emb: np.ndarray):
        self._emb = emb

    def __call__(self, waveform):
        return self._emb.reshape(1, -1)


class FakePipeline:
    """Pipeline que devuelve output con turnos + embeddings por audio."""

    def __init__(
        self,
        diarization_by_audio: dict[str, list[tuple[float, float, str]]],
        embeddings_by_audio: dict[str, dict[str, list[float]]] | None = None,
        ref_embedding: list[float] | None = None,
    ):
        self._diar = diarization_by_audio
        self._embs = embeddings_by_audio or {}
        self._ref_emb = np.array(ref_embedding or [1.0, 0.0, 0.0])
        # El embedding callable simula pipeline._inferences["_embedding"]
        self._inferences = {
            "_embedding": FakeEmbCallable(self._ref_emb),
        }

    def __call__(self, audio_path, **kwargs) -> FakePipelineOutput:
        key = str(audio_path)
        raw_tracks = self._diar.get(key, [])
        tracks = [
            (FakeSegment(s, e), f"track_{i}", label)
            for i, (s, e, label) in enumerate(raw_tracks)
        ]
        excl = FakeExclusiveDiarization(tracks)
        dia = FakeDiarization(tracks)

        # Embeddings
        embs_dict = self._embs.get(key, {})
        labels_sorted = sorted(embs_dict.keys()) if embs_dict else []
        if labels_sorted:
            embs_array = np.array([embs_dict[lbl] for lbl in labels_sorted])
        else:
            embs_array = None

        return FakePipelineOutput(excl, dia, embs_array)


class FakeAudioHelper:
    """Simula pyannote Audio con get_duration y crop."""

    def __init__(self, durations: dict[str, float]):
        self._durations = durations

    def get_duration(self, file_path) -> float:
        return self._durations.get(str(file_path), 30.0)

    def crop(self, file_path, segment):
        # Devuelve waveform dummy de 1xN muestras
        import torch

        dur = segment.end - segment.start
        n = int(dur * 16000)
        return torch.randn(1, n), 16000


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


def _make_pipeline(
    svc: DiarizacionService,
    playlist: str,
    video_tracks: dict[str, list[tuple[float, float, str]]],
    video_embs: dict[str, dict[str, list[float]]],
    ref_emb: list[float] | None = None,
) -> FakePipeline:
    """Construye un FakePipeline con las rutas correctas."""
    diar_by_audio = {}
    for vid, tracks in video_tracks.items():
        path = str(svc._audio_path(vid, playlist))
        diar_by_audio[path] = tracks

    embs_by_audio = {}
    for vid, embs in video_embs.items():
        path = str(svc._audio_path(vid, playlist))
        embs_by_audio[path] = embs

    return FakePipeline(diar_by_audio, embs_by_audio, ref_emb or [1.0, 0.0, 0.0])


def _patch_service(svc, pipeline, playlist, ref_duration=30.0):
    """Parchea el service con fakes."""
    ref_path = str(svc._ref_audio_path(playlist))
    return (
        patch.object(svc, "_get_pipeline", return_value=pipeline),
        patch.object(svc, "_get_audio_helper", return_value=FakeAudioHelper({
            ref_path: ref_duration,
        })),
    )


# ──────────────────────────────────────────────────────────
# Tests: constructor y defaults
# ──────────────────────────────────────────────────────────


class TestConstructor:
    """DiarizacionService defaults y encapsulación de config."""

    def test_defaults(self):
        svc = DiarizacionService()
        assert svc.actor == "Lilly Téllez"
        assert svc.video_ref_id == "su9nURIj9XQ"
        assert svc.umbral_match == pytest.approx(0.5)
        assert svc.umbral_ambiguo == pytest.approx(0.7)
        assert svc.pipeline_name == "pyannote/speaker-diarization-community-1"
        assert svc.fallback_pipeline == "pyannote-community/speaker-diarization-community-1"
        assert svc.device == "auto"

    def test_config_personalizada(self):
        svc = DiarizacionService(
            actor="X",
            video_ref_id="abc",
            umbral_match=0.3,
            umbral_ambiguo=0.6,
        )
        assert svc.actor == "X"
        assert svc.video_ref_id == "abc"
        assert svc.umbral_match == pytest.approx(0.3)

    def test_sin_embedding_model_param(self):
        """El constructor ya no acepta embedding_model (embedding interno del pipeline)."""
        svc = DiarizacionService()
        assert not hasattr(svc, "embedding_model")

    def test_get_pipeline_usa_adapter_primary_fallback_y_device(self, monkeypatch):
        calls = []

        def fake_load_pyannote_pipeline(**kwargs):
            calls.append(kwargs)
            return LoadedPyannotePipeline(
                pipeline="PIPELINE",
                pipeline_name="fallback-model",
                used_fallback=True,
            )

        monkeypatch.setattr(
            service_module,
            "load_pyannote_pipeline",
            fake_load_pyannote_pipeline,
        )
        monkeypatch.setattr(service_module, "_leer_token_hf", lambda: FAKE_CREDENTIAL)

        svc = DiarizacionService(
            pipeline_name="primary-model",
            fallback_pipeline="fallback-model",
            device="auto",
        )

        pipeline = svc._get_pipeline()

        assert pipeline == "PIPELINE"
        assert svc.pipeline_name == "fallback-model"
        assert calls == [
            {
                "primary_pipeline": "primary-model",
                "fallback_pipeline": "fallback-model",
                "token": FAKE_CREDENTIAL,
                "device": "auto",
            }
        ]


# ──────────────────────────────────────────────────────────
# Tests: procesar — un video, un speaker aceptado
# ──────────────────────────────────────────────────────────


class TestProcesarUnVideo:
    """procesar() con un video donde el actor habla."""

    def test_un_video_actor_identificado(self):
        svc = DiarizacionService()
        playlist = "test"

        transcript = _transcript([
            _seg("Hola", 0.0, 2.0),
            _seg("Mundo", 4.0, 6.0),
        ])

        pipeline = _make_pipeline(
            svc, playlist,
            video_tracks={"v1": [(0.0, 3.0, "SPEAKER_00"), (4.0, 7.0, "SPEAKER_00")]},
            video_embs={"v1": {"SPEAKER_00": [0.99, 0.01, 0.0]}},
        )

        p1, p2 = _patch_service(svc, pipeline, playlist)
        with p1, p2:
            resultado = svc.procesar([transcript], nombre_playlist=playlist)

        assert len(resultado) == 1
        assert resultado[0].video_id == "v1"
        assert len(resultado[0].raw_segments) == 2

    def test_un_video_actor_no_identificado(self):
        svc = DiarizacionService()
        playlist = "test"

        transcript = _transcript([_seg("Hola", 0.0, 2.0)])

        pipeline = _make_pipeline(
            svc, playlist,
            video_tracks={"v1": [(0.0, 3.0, "SPEAKER_00")]},
            video_embs={"v1": {"SPEAKER_00": [0.0, 1.0, 0.0]}},  # rechazado
        )

        p1, p2 = _patch_service(svc, pipeline, playlist)
        with p1, p2:
            resultado = svc.procesar([transcript], nombre_playlist=playlist)

        assert len(resultado) == 1
        assert len(resultado[0].raw_segments) == 0


# ──────────────────────────────────────────────────────────
# Tests: múltiples videos
# ──────────────────────────────────────────────────────────


class TestProcesarMultiplesVideos:
    """procesar() itera videos independientemente."""

    def test_dos_videos_actor_en_ambos(self):
        svc = DiarizacionService()
        playlist = "test"

        t1 = _transcript([_seg("A", 0.0, 1.0)], video_id="v1")
        t2 = _transcript([_seg("B", 0.0, 1.0)], video_id="v2")

        pipeline = _make_pipeline(
            svc, playlist,
            video_tracks={
                "v1": [(0.0, 2.0, "SPEAKER_00")],
                "v2": [(0.0, 2.0, "SPEAKER_01")],
            },
            video_embs={
                "v1": {"SPEAKER_00": [0.99, 0.01, 0.0]},
                "v2": {"SPEAKER_01": [0.98, 0.02, 0.0]},
            },
        )

        p1, p2 = _patch_service(svc, pipeline, playlist)
        with p1, p2:
            resultado = svc.procesar([t1, t2], nombre_playlist=playlist)

        assert len(resultado) == 2
        assert len(resultado[0].raw_segments) == 1
        assert len(resultado[1].raw_segments) == 1


# ──────────────────────────────────────────────────────────
# Tests: edge cases
# ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """procesar() con entradas vacías."""

    def test_lista_vacia(self):
        svc = DiarizacionService()
        pipeline = FakePipeline({})

        p1, p2 = _patch_service(svc, pipeline, "test")
        with p1, p2:
            resultado = svc.procesar([], nombre_playlist="test")

        assert resultado == []

    def test_video_sin_turnos(self):
        """Video donde el pipeline no detecta turnos."""
        svc = DiarizacionService()
        playlist = "test"

        transcript = _transcript([_seg("Hola", 0.0, 2.0)], video_id="v1")

        pipeline = _make_pipeline(
            svc, playlist,
            video_tracks={"v1": []},  # sin turnos
            video_embs={"v1": {}},
        )

        p1, p2 = _patch_service(svc, pipeline, playlist)
        with p1, p2:
            resultado = svc.procesar([transcript], nombre_playlist=playlist)

        assert len(resultado) == 1
        assert len(resultado[0].raw_segments) == 0
