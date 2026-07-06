"""Construye el PerfilVozActor desde un audio de referencia.

Función pura: recibe el pipeline de embedding ya cargado y un audio_helper
para medir duración. Extrae un embedding del audio completo (window="whole")
y devuelve un PerfilVozActor con cache solo en memoria.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import PerfilVozActor

logger = logging.getLogger(__name__)


def construir_perfil(
    audio_ref: Path | str,
    actor: str,
    video_id_ref: str,
    modelo_embedding: str,
    embedding_pipeline,
    audio_helper,
) -> PerfilVozActor:
    """Extrae el embedding de voz del actor desde el audio de referencia.

    Usa el pipeline de embedding sobre el audio completo (asume que el audio
    de referencia es mayoritariamente del actor objetivo).

    Args:
        audio_ref: Ruta al .wav de referencia del actor.
        actor: Nombre del actor político objetivo.
        video_id_ref: ID del video de YouTube usado como referencia.
        modelo_embedding: Nombre del modelo de embedding usado.
        embedding_pipeline: Pipeline callable que devuelve (1, D) ndarray.
        audio_helper: Instancia con .get_duration(file) → float.

    Returns:
        PerfilVozActor con embedding promedio del audio de referencia.
    """
    duracion = audio_helper.get_duration(str(audio_ref))

    emb = embedding_pipeline(str(audio_ref))
    emb_row = emb[0]
    embedding = emb_row.tolist() if hasattr(emb_row, "tolist") else list(emb_row)

    logger.info(
        f"Perfil de voz construido: actor='{actor}', "
        f"dim={len(embedding)}, duración={duracion:.1f}s"
    )

    return PerfilVozActor(
        actor=actor,
        video_id_referencia=video_id_ref,
        embedding=embedding,
        modelo_embedding=modelo_embedding,
        duracion_segundos=duracion,
    )
