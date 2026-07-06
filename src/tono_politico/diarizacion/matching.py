"""Matching de speakers contra el perfil de voz del actor objetivo.

Tres funciones puras:
    distancia_coseno(a, b)        → float
    clasificar_speaker(...)        → SpeakerMatch
    identificar_actor(dict, perfil) → list[SpeakerMatch]

Los thresholds por defecto (0.5 / 0.7) están calibrados con base en
la literatura de pyannote/embedding + VoxCeleb (ver docs de diseño).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

from .models import PerfilVozActor, SpeakerMatch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Distancia coseno
# ──────────────────────────────────────────────────────────


def distancia_coseno(a: Sequence[float], b: Sequence[float]) -> float:
    """Distancia coseno entre dos vectores: 1 - similitud_coseno.

    Rango:
        0.0 = vectores idénticos (mismo speaker)
        1.0 = ortogonales (sin relación)
        2.0 = opuestos

    Args:
        a: Primer vector de embedding.
        b: Segundo vector de embedding.

    Returns:
        Distancia coseno en [0.0, 2.0].
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 2.0

    return 1.0 - (dot / (norm_a * norm_b))


# ──────────────────────────────────────────────────────────
# Clasificación de un speaker
# ──────────────────────────────────────────────────────────


def clasificar_speaker(
    speaker_id: str,
    distancia: float,
    umbral_match: float = 0.5,
    umbral_ambiguo: float = 0.7,
) -> SpeakerMatch:
    """Clasifica un speaker según su distancia coseno al perfil del actor.

    Reglas (fronteras exclusivas):
        distancia < umbral_match           → aceptado
        umbral_match ≤ distancia < umbral_ambiguo → ambiguo (descartar)
        distancia ≥ umbral_ambiguo          → rechazado

    Args:
        speaker_id: Etiqueta del speaker (SPEAKER_00, ...).
        distancia: Distancia coseno al perfil del actor.
        umbral_match: Distancia por debajo de la cual se acepta.
        umbral_ambiguo: Distancia por encima de la cual se rechaza.

    Returns:
        SpeakerMatch con la clasificación correspondiente.
    """
    if distancia < umbral_match:
        return SpeakerMatch(
            speaker_id=speaker_id,
            distancia=distancia,
            aceptado=True,
            es_ambiguo=False,
        )
    elif distancia < umbral_ambiguo:
        return SpeakerMatch(
            speaker_id=speaker_id,
            distancia=distancia,
            aceptado=False,
            es_ambiguo=True,
        )
    else:
        return SpeakerMatch(
            speaker_id=speaker_id,
            distancia=distancia,
            aceptado=False,
            es_ambiguo=False,
        )


# ──────────────────────────────────────────────────────────
# Identificación del actor
# ──────────────────────────────────────────────────────────


def identificar_actor(
    speaker_embeddings: dict[str, list[float]],
    perfil: PerfilVozActor,
    umbral_match: float = 0.5,
    umbral_ambiguo: float = 0.7,
) -> list[SpeakerMatch]:
    """Compara cada speaker diarizado contra el perfil del actor objetivo.

    Args:
        speaker_embeddings: {speaker_id: embedding_promedio} para un video.
        perfil: PerfilVozActor con el embedding de referencia del actor.
        umbral_match: Distancia por debajo de la cual se acepta.
        umbral_ambiguo: Distancia por encima de la cual se rechaza.

    Returns:
        Lista de SpeakerMatch ordenada por distancia ascendente.
    """
    resultados: list[SpeakerMatch] = []

    for speaker_id, emb in speaker_embeddings.items():
        dist = distancia_coseno(perfil.embedding, emb)
        match = clasificar_speaker(
            speaker_id=speaker_id,
            distancia=dist,
            umbral_match=umbral_match,
            umbral_ambiguo=umbral_ambiguo,
        )
        resultados.append(match)

    resultados.sort(key=lambda m: m.distancia)

    aceptados = [m for m in resultados if m.aceptado]
    ambiguos = [m for m in resultados if m.es_ambiguo]
    logger.info(
        f"Matching: {len(aceptados)} aceptado(s), "
        f"{len(ambiguos)} ambiguo(s), "
        f"{len(resultados) - len(aceptados) - len(ambiguos)} rechazado(s)"
    )

    return resultados
