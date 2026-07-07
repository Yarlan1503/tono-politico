"""Alineación de transcripciones con turnos del actor objetivo.

Función pura: recibe un VideoTranscript y los turnos del actor (ya
identificados), conserva solo los SegmentoRaw cuyo midpoint temporal
cae dentro de algún turno del actor.

Criterio: midpoint del segmento = (t_start + t_end) / 2.
    Inclusivo en t_start del turno, exclusivo en t_end.
    Turnos de otro video_id se ignoran.
"""

from __future__ import annotations

import bisect
import logging

from ..models import VideoTranscript
from .models import TurnoOrador

logger = logging.getLogger(__name__)


def filtrar_por_actor(
    transcript: VideoTranscript,
    turnos_actor: list[TurnoOrador],
) -> VideoTranscript:
    """Filtra un VideoTranscript conservando solo segmentos del actor.

    Para cada SegmentoRaw, calcula el midpoint temporal y usa ``bisect``
    sobre un índice ordenado de turnos para verificar coincidencia en
    O(log M) por segmento, en vez de búsqueda lineal.

    Inclusivo en ``t_start`` del turno, exclusivo en ``t_end``.
    Turnos de otro ``video_id`` se ignoran.
    """
    # Filtrar y ordenar turnos del mismo video por t_start
    rangos = sorted(
        ((t.t_start, t.t_end) for t in turnos_actor if t.video_id == transcript.video_id),
        key=lambda r: r[0],
    )

    if not rangos:
        logger.info(f"Sin turnos del actor en video {transcript.video_id}, 0 segmentos conservados")
        return VideoTranscript(
            video_id=transcript.video_id,
            url=transcript.url,
            titulo=transcript.titulo,
            fecha=transcript.fecha,
            raw_segments=[],
        )

    # Construir índice de starts para bisect
    starts = [r[0] for r in rangos]

    conservados = [
        seg
        for seg in transcript.raw_segments
        if _midpoint_en_rangos_bisect(seg.t_start, seg.t_end, starts, rangos)
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


def _midpoint_en_rangos_bisect(
    t_start: float,
    t_end: float,
    starts: list[float],
    rangos: list[tuple[float, float]],
) -> bool:
    """Verifica si el midpoint de [t_start, t_end] cae en algún rango.

    Usa ``bisect_right`` para encontrar el último turno cuyo ``t_start``
    es <= midpoint, luego verifica ``midpoint < t_end``.

    Inclusivo en el inicio del rango, exclusivo en el fin.
    """
    midpoint = (t_start + t_end) / 2.0

    # bisect_right encuentra la posición de inserción después de todos
    # los starts <= midpoint. Restamos 1 para obtener el último candidato.
    idx = bisect.bisect_right(starts, midpoint) - 1
    if idx < 0:
        return False

    r_start, r_end = rangos[idx]
    return r_start <= midpoint < r_end
