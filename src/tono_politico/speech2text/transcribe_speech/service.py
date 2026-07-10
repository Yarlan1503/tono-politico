"""TranscribeSpeechService — Whisper sobre turnos del actor.

Entrada: AudioVideo + turnos del actor (pyannote).
Salida: ActorTranscript turn-level (actor_transcript.v1).

No persiste word-level, probability ni verbose Whisper.
"""

from __future__ import annotations

import logging
from typing import Any

from tono_politico.speech2text.audio_fetcher.models import AudioVideo

from ..models import ActorTranscript, TurnoOrador
from .actor_clip import transcribir_turnos_actor
from .transcription_clip import WhisperFfmpegClipTranscriber

logger = logging.getLogger(__name__)


class TranscribeSpeechService:
    """ASR actor-only a partir de turnos diarizados."""

    def __init__(
        self,
        actor: str = "Lilly Téllez",
        whisper_model: str = "large-v3-turbo",
        idioma: str = "es",
        padding_segundos: float = 0.0,
        transcriptor: Any | None = None,
    ) -> None:
        self.actor = actor
        self.whisper_model = whisper_model
        self.idioma = idioma
        self.padding_segundos = padding_segundos
        self._transcriptor = transcriptor

    def procesar_one(
        self,
        audio: AudioVideo,
        turnos_actor: list[TurnoOrador],
    ) -> ActorTranscript | None:
        """Transcribe solo los turnos del actor.

        Returns:
            ``ActorTranscript`` si hay segmentos con texto; ``None`` si no hay
            turnos o el ASR no produjo texto.
        """
        if not turnos_actor:
            logger.info("Video %s: sin turnos del actor para ASR", audio.video_id)
            return None

        transcript = transcribir_turnos_actor(
            audio.audio_path,
            turnos_actor,
            video_id=audio.video_id,
            actor=self.actor,
            transcriptor=self._get_transcriptor(),
            modelo=self.whisper_model,
            idioma=self.idioma,
            padding_segundos=self.padding_segundos,
            duracion_audio=audio.duracion if audio.duracion > 0 else None,
            fecha=audio.fecha,
        )
        if not transcript.segments:
            logger.info("Video %s: ASR sin segmentos con texto", audio.video_id)
            return None
        if transcript.fecha is None and audio.fecha is not None:
            # defensa en profundidad si el builder omitiera fecha
            transcript.fecha = audio.fecha
        return transcript

    def _get_transcriptor(self) -> Any:
        if self._transcriptor is None:
            self._transcriptor = WhisperFfmpegClipTranscriber()
        return self._transcriptor
