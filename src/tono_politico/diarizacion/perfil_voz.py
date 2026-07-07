"""Construye el PerfilVozActor desde un audio de referencia.

Función pura: recibe un extractor/callable de embedding ya cargado y un audio_helper
para medir duración. Extrae un embedding del audio completo (window="whole")
y devuelve un PerfilVozActor con cache solo en memoria.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import PerfilVozActor

logger = logging.getLogger(__name__)


def construir_perfil_desde_output(
    output,
    actor: str,
    video_ref_id: str,
    pipeline_name: str,
) -> PerfilVozActor:
    """Construye PerfilVozActor desde output.speaker_embeddings público.

    Ejecuta el pipeline sobre el audio de referencia y selecciona el
    embedding del speaker con mayor duración total.

    Args:
        output: Salida del pipeline de diarización sobre el audio de referencia.
        actor: Nombre del actor político objetivo.
        video_ref_id: ID del video de referencia.
        pipeline_name: Nombre del pipeline usado (para metadata).

    Returns:
        PerfilVozActor con el embedding del speaker dominante.

    Raises:
        ValueError: Si no hay embeddings en el output.
    """
    import numpy as np

    embs = output.speaker_embeddings
    if embs is None or len(embs) == 0:
        raise ValueError(
            "El output del pipeline no contiene speaker_embeddings. "
            "Verifica que el audio de referencia tenga contenido de voz."
        )

    labels = list(output.speaker_diarization.labels())
    if not labels:
        raise ValueError("El output del pipeline no contiene speakers diarizados.")

    # Calcular duración total por speaker desde exclusive_speaker_diarization
    duraciones: dict[str, float] = {label: 0.0 for label in labels}
    for segment, _track, label in output.exclusive_speaker_diarization.itertracks(yield_label=True):
        if label in duraciones:
            duraciones[label] += segment.end - segment.start

    # Speaker dominante = mayor duración
    speaker_dominante = max(duraciones, key=lambda k: duraciones[k])
    idx = labels.index(speaker_dominante)

    emb = np.array(embs[idx]).flatten()
    duracion_total = duraciones[speaker_dominante]

    logger.info(
        f"Perfil de voz construido: actor='{actor}', dim={len(emb)}, "
        f"speaker_dominante='{speaker_dominante}', duración={duracion_total:.1f}s"
    )

    return PerfilVozActor(
        actor=actor,
        video_id_referencia=video_ref_id,
        embedding=emb.tolist(),
        modelo_embedding=f"speaker_embeddings:{pipeline_name}",
        duracion_segundos=duracion_total,
    )


def construir_perfil(
    audio_ref: Path | str,
    actor: str,
    video_id_ref: str,
    modelo_embedding: str,
    embedding_pipeline,
    audio_helper,
) -> PerfilVozActor:
    """Extrae el embedding de voz del actor desde el audio de referencia.

    Usa el callable de embedding sobre el audio completo (asume que el audio
    de referencia es mayoritariamente del actor objetivo).

    Args:
        audio_ref: Ruta al .wav de referencia del actor.
        actor: Nombre del actor político objetivo.
        video_id_ref: ID del video de YouTube usado como referencia.
        modelo_embedding: Identificador del origen del embedding usado.
        embedding_pipeline: Callable que devuelve (1, D) ndarray/list.
        audio_helper: Instancia con .get_duration(file) → float.

    Returns:
        PerfilVozActor con embedding promedio del audio de referencia.
    """
    duracion = audio_helper.get_duration(str(audio_ref))

    emb = embedding_pipeline(str(audio_ref))
    emb_row = emb[0]
    embedding = emb_row.tolist() if hasattr(emb_row, "tolist") else list(emb_row)

    logger.info(
        f"Perfil de voz construido: actor='{actor}', dim={len(embedding)}, duración={duracion:.1f}s"
    )

    return PerfilVozActor(
        actor=actor,
        video_id_referencia=video_id_ref,
        embedding=embedding,
        modelo_embedding=modelo_embedding,
        duracion_segundos=duracion,
    )
