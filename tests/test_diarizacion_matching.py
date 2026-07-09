"""Tests para matching.py: distancia coseno, clasificación e identificación de actor."""

from __future__ import annotations

import pytest

from tono_politico.speech2text.diarization.matching import (
    clasificar_speaker,
    distancia_coseno,
    identificar_actor,
)
from tono_politico.speech2text.diarization.models import PerfilVozActor

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────


def _perfil(embedding: list[float]) -> PerfilVozActor:
    return PerfilVozActor(
        actor="X",
        video_id_referencia="ref",
        embedding=embedding,
        modelo_embedding="test",
        duracion_segundos=10.0,
    )


# ──────────────────────────────────────────────────────────
# distancia_coseno
# ──────────────────────────────────────────────────────────


class TestDistanciaCoseno:
    """distancia_coseno(): pure math entre dos vectores."""

    def test_vectores_identicos(self):
        """Dos vectores idénticos → distancia 0.0."""
        v = [1.0, 0.5, -0.3]
        assert distancia_coseno(v, v) == pytest.approx(0.0)

    def test_vectores_ortogonales(self):
        """Vectores ortogonales → distancia 1.0."""
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert distancia_coseno(a, b) == pytest.approx(1.0)

    def test_vectores_opuestos(self):
        """Vectores opuestos → distancia 2.0."""
        a = [1.0, 1.0]
        b = [-1.0, -1.0]
        assert distancia_coseno(a, b) == pytest.approx(2.0)

    def test_similitud_alta(self):
        """Vectores muy similares → distancia baja (< 0.5)."""
        a = [1.0, 0.1, 0.05]
        b = [1.0, 0.12, 0.04]
        d = distancia_coseno(a, b)
        assert d < 0.5

    def test_dim_alta(self):
        """Funciona con dimensiones altas (ej. embeddings de speaker)."""
        import random

        random.seed(42)
        a = [random.gauss(0, 1) for _ in range(512)]
        b = a.copy()  # idéntico
        assert distancia_coseno(a, b) == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────
# clasificar_speaker
# ──────────────────────────────────────────────────────────


class TestClasificarSpeaker:
    """clasificar_speaker(): umbral_match y umbral_ambiguo → SpeakerMatch."""

    def test_aceptado_claro(self):
        """Distancia < umbral_match → aceptado, no ambiguo."""
        m = clasificar_speaker("SPEAKER_00", 0.3, umbral_match=0.5, umbral_ambiguo=0.7)
        assert m.aceptado is True
        assert m.es_ambiguo is False

    def test_ambiguo(self):
        """Distancia entre umbral_match y umbral_ambiguo → no aceptado, ambiguo."""
        m = clasificar_speaker("SPEAKER_01", 0.6, umbral_match=0.5, umbral_ambiguo=0.7)
        assert m.aceptado is False
        assert m.es_ambiguo is True

    def test_rechazado_claro(self):
        """Distancia > umbral_ambiguo → no aceptado, no ambiguo."""
        m = clasificar_speaker("SPEAKER_02", 0.9, umbral_match=0.5, umbral_ambiguo=0.7)
        assert m.aceptado is False
        assert m.es_ambiguo is False

    def test_exactamente_en_umbral_match(self):
        """Distancia == umbral_match → ambiguo (frontera exclusiva)."""
        m = clasificar_speaker("SPEAKER_00", 0.5, umbral_match=0.5, umbral_ambiguo=0.7)
        assert m.aceptado is False
        assert m.es_ambiguo is True

    def test_exactamente_en_umbral_ambiguo(self):
        """Distancia == umbral_ambiguo → rechazado (frontera exclusiva)."""
        m = clasificar_speaker("SPEAKER_00", 0.7, umbral_match=0.5, umbral_ambiguo=0.7)
        assert m.aceptado is False
        assert m.es_ambiguo is False

    def test_speaker_id_se_propaga(self):
        """El speaker_id se conserva en el resultado."""
        m = clasificar_speaker("SPEAKER_99", 0.1, umbral_match=0.5, umbral_ambiguo=0.7)
        assert m.speaker_id == "SPEAKER_99"


# ──────────────────────────────────────────────────────────
# identificar_actor
# ──────────────────────────────────────────────────────────


class TestIdentificarActor:
    """identificar_actor(): dict de embeddings + perfil → list[SpeakerMatch]."""

    def test_un_speaker_aceptado(self):
        """Un speaker que matchea claramente → un SpeakerMatch aceptado."""
        emb_actor = [1.0, 0.0, 0.0]
        emb_speaker = [0.99, 0.01, 0.0]

        perfil = _perfil(emb_actor)
        resultados = identificar_actor(
            {"SPEAKER_00": emb_speaker},
            perfil,
            umbral_match=0.5,
            umbral_ambiguo=0.7,
        )

        assert len(resultados) == 1
        assert resultados[0].speaker_id == "SPEAKER_00"
        assert resultados[0].aceptado is True
        assert resultados[0].es_ambiguo is False

    def test_tres_speakers_mezcla(self):
        """Tres speakers: uno aceptado, uno ambiguo, uno rechazado."""
        emb_actor = [1.0, 0.0, 0.0]
        emb_cercano = [0.99, 0.01, 0.0]  # distancia ~0.0 → aceptado
        emb_medio = [0.4, 0.85, 0.1]  # distancia ~0.58 → ambiguo
        emb_lejano = [0.0, 1.0, 0.0]  # distancia 1.0 → rechazado

        perfil = _perfil(emb_actor)
        resultados = identificar_actor(
            {
                "SPEAKER_00": emb_cercano,
                "SPEAKER_01": emb_medio,
                "SPEAKER_02": emb_lejano,
            },
            perfil,
            umbral_match=0.5,
            umbral_ambiguo=0.7,
        )

        assert len(resultados) == 3
        by_speaker = {r.speaker_id: r for r in resultados}
        assert by_speaker["SPEAKER_00"].aceptado is True
        assert by_speaker["SPEAKER_01"].aceptado is False
        assert by_speaker["SPEAKER_02"].aceptado is False

    def test_ningun_match_todos_rechazados(self):
        """Ningún speaker matchea → todos rechazados, ninguno aceptado."""
        emb_actor = [1.0, 0.0, 0.0]
        emb_lejano_1 = [0.0, 1.0, 0.0]
        emb_lejano_2 = [0.0, 0.0, 1.0]

        perfil = _perfil(emb_actor)
        resultados = identificar_actor(
            {"SPEAKER_00": emb_lejano_1, "SPEAKER_01": emb_lejano_2},
            perfil,
            umbral_match=0.5,
            umbral_ambiguo=0.7,
        )

        assert len(resultados) == 2
        assert all(not r.aceptado for r in resultados)
        assert all(not r.es_ambiguo for r in resultados)

    def test_todos_ambiguos(self):
        """Todos los speakers caen en zona ambigua → ninguno aceptado."""
        emb_actor = [1.0, 0.0]
        # 67° → distancia coseno ≈ 0.61 (entre 0.5 y 0.7)
        import math

        angulo = math.radians(67)
        emb = [math.cos(angulo), math.sin(angulo)]

        perfil = _perfil(emb_actor)
        resultados = identificar_actor(
            {"SPEAKER_00": emb},
            perfil,
            umbral_match=0.5,
            umbral_ambiguo=0.7,
        )

        assert len(resultados) == 1
        assert resultados[0].aceptado is False
        assert resultados[0].es_ambiguo is True

    def test_orden_por_distancia(self):
        """Los resultados se ordenan por distancia ascendente."""
        emb_actor = [1.0, 0.0, 0.0]

        perfil = _perfil(emb_actor)
        resultados = identificar_actor(
            {
                "SPEAKER_02": [0.0, 1.0, 0.0],  # lejos
                "SPEAKER_00": [0.99, 0.01, 0.0],  # cerca
                "SPEAKER_01": [0.5, 0.5, 0.0],  # medio
            },
            perfil,
            umbral_match=0.5,
            umbral_ambiguo=0.7,
        )

        distancias = [r.distancia for r in resultados]
        assert distancias == sorted(distancias)
        assert resultados[0].speaker_id == "SPEAKER_00"

    def test_speaker_embeddings_vacio(self):
        """Dict vacío → lista vacía."""
        perfil = _perfil([1.0, 0.0])
        resultados = identificar_actor({}, perfil)
        assert resultados == []

    def test_defaults_05_07(self):
        """Los defaults umbral_match=0.5 y umbral_ambiguo=0.7 se aplican."""
        emb_actor = [1.0, 0.0]
        emb_cercano = [0.99, 0.01]

        perfil = _perfil(emb_actor)
        resultados = identificar_actor(
            {"SPEAKER_00": emb_cercano},
            perfil,
        )

        assert resultados[0].aceptado is True
