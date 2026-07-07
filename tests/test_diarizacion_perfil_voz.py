"""Tests para construir_perfil(): extrae embedding del audio de referencia."""

from __future__ import annotations

from pathlib import Path

from tono_politico.diarizacion.models import PerfilVozActor
from tono_politico.diarizacion.perfil_voz import construir_perfil

# ──────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────


class FakeAudio:
    """Simula pyannote Audio para medir duración."""

    def __init__(self, duration: float):
        self._duration = duration

    def get_duration(self, file_path) -> float:
        return self._duration


class FakeEmbeddingPipeline:
    """Simula extractor de embedding — devuelve (1, D) array."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.calls: list[str] = []

    def __call__(self, audio_path) -> FakeEmbeddingPipeline:
        self.calls.append(str(audio_path))
        return self

    @property
    def shape(self):
        return (1, self.dim)

    def __getitem__(self, idx):
        """Simula ndarray indexing para emb[0] → list[float]."""
        if idx == 0:
            return [0.1 * i for i in range(self.dim)]
        raise IndexError

    def tolist(self):
        return [[0.1 * i for i in range(self.dim)]]


# ──────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────


class TestConstruirPerfil:
    """construir_perfil(): audio ref + modelo → PerfilVozActor."""

    def test_perfil_basico(self):
        """Construye un perfil válido desde un audio de referencia."""
        audio_ref = Path("/fake/lilly_ref.wav")
        pipeline = FakeEmbeddingPipeline(dim=16)
        audio_helper = FakeAudio(duration=30.0)

        perfil = construir_perfil(
            audio_ref=audio_ref,
            actor="Lilly Téllez",
            video_id_ref="su9nURIj9XQ",
            modelo_embedding="pipeline-internal",
            embedding_pipeline=pipeline,
            audio_helper=audio_helper,
        )

        assert isinstance(perfil, PerfilVozActor)
        assert perfil.actor == "Lilly Téllez"
        assert perfil.video_id_referencia == "su9nURIj9XQ"
        assert perfil.modelo_embedding == "pipeline-internal"
        assert len(perfil.embedding) == 16
        assert all(isinstance(x, float) for x in perfil.embedding)
        assert perfil.duracion_segundos == 30.0

    def test_embedding_es_1d_no_2d(self):
        """El embedding almacenado es 1D (list[float]), no (1, D)."""
        pipeline = FakeEmbeddingPipeline(dim=8)
        audio_helper = FakeAudio(duration=15.0)

        perfil = construir_perfil(
            audio_ref=Path("/fake/v.wav"),
            actor="X",
            video_id_ref="v1",
            modelo_embedding="m",
            embedding_pipeline=pipeline,
            audio_helper=audio_helper,
        )

        # Debe ser lista plana de floats, no lista de listas
        assert isinstance(perfil.embedding, list)
        assert not isinstance(perfil.embedding[0], list)
        assert len(perfil.embedding) == 8

    def test_pipeline_se_llama_una_vez(self):
        """El pipeline se ejecuta exactamente una vez sobre el audio."""
        pipeline = FakeEmbeddingPipeline(dim=4)
        audio_helper = FakeAudio(duration=10.0)

        construir_perfil(
            audio_ref=Path("/fake/once.wav"),
            actor="A",
            video_id_ref="v",
            modelo_embedding="m",
            embedding_pipeline=pipeline,
            audio_helper=audio_helper,
        )

        assert len(pipeline.calls) == 1
        assert pipeline.calls[0] == "/fake/once.wav"

    def test_duracion_se_propaga(self):
        """La duración reportada viene del audio_helper."""
        pipeline = FakeEmbeddingPipeline(dim=4)

        perfil = construir_perfil(
            audio_ref=Path("/fake/long.wav"),
            actor="A",
            video_id_ref="v",
            modelo_embedding="m",
            embedding_pipeline=pipeline,
            audio_helper=FakeAudio(duration=120.5),
        )

        assert perfil.duracion_segundos == 120.5
