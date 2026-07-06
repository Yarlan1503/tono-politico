"""Alineación de transcripciones con turnos del actor objetivo.

Función pura: recibe un VideoTranscript y los turnos del actor (ya
identificados), conserva solo los SegmentoRaw cuyo midpoint temporal
cae dentro de algún turno del actor.

Criterio: midpoint del segmento = (t_start + t_end) / 2.
    Inclusivo en t_start del turno, exclusivo en t_end.
    Turnos de otro video_id se ignoran.
"""

from __future__ import annotations

import logging

from ..models import VideoTranscript
from .models import TurnoOrador

logger = logging.getLogger(__name__)


def filtrar_por_actor(
    transcript: VideoTranscript,
    turnos_actor: list[TurnoOrador],
) -> VideoTranscript:
    """Filtra un VideoTranscript conservando solo segmentos del actor.

    Para cada SegmentoRaw, calcula el midpoint temporal y verifica si
    cae dentro de algún turno del actor en el mismo video. Solo se
    conservan los segmentos que coinciden.

    Args:
        transcript: VideoTranscript con todos los segmentos del video.
        turnos_actor: Turnos del actor objetivo (pueden incluir otros
            videos; se filtran por video_id).

    Returns:
        Nuevo VideoTranscript con metadata preservada y solo los
        segmentos atribuidos al actor.
    """
    # Filtrar turnos del mismo video
    rangos = [
        (t.t_start, t.t_end)
        for t in turnos_actor
        if t.video_id == transcript.video_id
    ]

    if not rangos:
        logger.info(
            f"Sin turnos del actor en video {transcript.video_id}, "
            f"0 segmentos conservados"
        )
        return VideoTranscript(
            video_id=transcript.video_id,
            url=transcript.url,
            titulo=transcript.titulo,
            fecha=transcript.fecha,
            raw_segments=[],
        )

    conservados = [
        seg for seg in transcript.raw_segments
        if _midpoint_en_rangos(seg.t_start, seg.t_end, rangos)
    ]

    logger.info(
        f"Video {transcript.video_id}: {len(conservados)}/"
        f"{len(transcript.raw_segments)} segmentos del actor"
    )

    return VideoTranscript(
        video_id=transcript.video_id,
        url=transcript.url,
        titulo=transcript.titulo,
        fecha=transcript.fecha,
        raw_segments=conservados,
    )


def _midpoint_en_rangos(
    t_start: float,
    t_end: float,
    rangos: list[tuple[float, float]],
) -> bool:
    """Verifica si el midpoint de [t_start, t_end] cae en algún rango.

    Inclusivo en el inicio del rango, exclusivo en el fin.
    """
    midpoint = (t_start + t_end) / 2.0
    for r_start, r_end in rangos:
        if r_start <= midpoint < r_end:
            return True
    return False
