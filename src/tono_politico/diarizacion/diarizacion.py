"""Ejecuta pyannote community-1 sobre un WAV y extrae turnos exclusivos.

Función pura: recibe un pipeline ya cargado (no lo instancia) y un path
de audio, devuelve lista de TurnoOrador usando exclusive_speaker_diarization
(sin traslapes) para alineación limpia con Whisper.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import TurnoOrador

logger = logging.getLogger(__name__)


def diarizar(
    audio_path: Path | str,
    pipeline,
    video_id: str,
) -> list[TurnoOrador]:
    """Ejecuta el pipeline de diarización sobre un audio y extrae turnos.

    Usa exclusive_speaker_diarization para obtener turnos sin traslapes,
    ideal para alinear con timestamps de Whisper.

    Args:
        audio_path: Ruta al archivo .wav del video.
        pipeline: Instancia de pyannote.audio.Pipeline ya cargada.
        video_id: ID del video de YouTube para etiquetar los turnos.

    Returns:
        Lista de TurnoOrador ordenada cronológicamente.
    """
    output = pipeline(str(audio_path))

    turnos: list[TurnoOrador] = []
    for segment, _track, speaker in (
        output.exclusive_speaker_diarization.itertracks(yield_label=True)
    ):
        turnos.append(
            TurnoOrador(
                video_id=video_id,
                speaker_id=speaker,
                t_start=segment.start,
                t_end=segment.end,
            )
        )

    logger.info(
        f"Diarización de {video_id}: {len(turnos)} turnos, "
        f"{len({t.speaker_id for t in turnos})} speakers"
    )
    return turnos
