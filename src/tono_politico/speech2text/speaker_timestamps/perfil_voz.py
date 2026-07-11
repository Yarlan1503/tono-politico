"""Construye el PerfilVozActor desde un audio de referencia.

Función pura: recibe un extractor/callable de embedding ya cargado y un audio_helper
para medir duración. Extrae un embedding del audio completo (window="whole")
y devuelve un PerfilVozActor con cache solo en memoria.
"""

from __future__ import annotations

import logging

from .models import PerfilVozActor

logger = logging.getLogger(__name__)


def construir_perfil_desde_output(
    output,
    actor: str,
    video_ref_id: str,
    pipeline_name: str,
) -> PerfilVozActor:
    """Construye un perfil desde embeddings y diarización ya ejecutadas."""
    import numpy as np

    embs = getattr(output, "speaker_embeddings", None)
    if embs is None or len(embs) == 0:
        raise ValueError(
            "El output del pipeline no contiene speaker_embeddings. "
            "Verifica que el audio de referencia tenga contenido de voz."
        )

    speaker_diarization = getattr(output, "speaker_diarization", None)
    exclusive_diarization = getattr(output, "exclusive_speaker_diarization", None)
    if speaker_diarization is None or not hasattr(speaker_diarization, "labels"):
        raise ValueError("El output del pipeline no contiene speakers diarizados.")
    if exclusive_diarization is None or not hasattr(exclusive_diarization, "itertracks"):
        raise ValueError("El output no contiene exclusive_speaker_diarization iterable.")

    labels = list(speaker_diarization.labels())
    if not labels:
        raise ValueError("El output del pipeline no contiene speakers diarizados.")
    if len(embs) != len(labels):
        raise ValueError(
            f"cantidad de embeddings inconsistente con labels: {len(embs)} != {len(labels)}"
        )

    embeddings: list[np.ndarray] = []
    for label, raw_embedding in zip(labels, embs, strict=True):
        if raw_embedding is None:
            raise ValueError(f"embedding ausente para speaker {label}")
        embedding = np.asarray(raw_embedding, dtype=float).reshape(-1)
        if embedding.size == 0 or not np.isfinite(embedding).all():
            raise ValueError(f"embedding inválido para speaker {label}")
        embeddings.append(embedding)

    duraciones: dict[str, float] = {str(label): 0.0 for label in labels}
    for segment, _track, label in exclusive_diarization.itertracks(yield_label=True):
        start = float(segment.start)
        end = float(segment.end)
        if start < 0 or end <= start:
            raise ValueError("segmento de diarización inválido: end debe ser mayor que start")
        if str(label) in duraciones:
            duraciones[str(label)] += end - start

    speaker_dominante = max(duraciones, key=lambda key: duraciones[key])
    idx = list(map(str, labels)).index(speaker_dominante)
    emb = embeddings[idx]
    duracion_total = duraciones[speaker_dominante]

    logger.info(
        "Perfil de voz construido: actor=%r dim=%s speaker_dominante=%r duración=%.1fs",
        actor,
        len(emb),
        speaker_dominante,
        duracion_total,
    )

    return PerfilVozActor(
        actor=actor,
        video_id_referencia=video_ref_id,
        embedding=emb.tolist(),
        modelo_embedding=f"speaker_embeddings:{pipeline_name}",
        duracion_segundos=duracion_total,
    )
