"""SpeechToTextService — orquesta audio_fetcher + speaker_timestamps + transcribe_speech."""

from __future__ import annotations

import logging
from pathlib import Path

from tono_politico.speech2text.audio_fetcher import AudioFetcherService, AudioVideo, VideoMeta
from tono_politico.speech2text.speaker_timestamps import SpeakerTimestampsService
from tono_politico.speech2text.transcribe_speech import TranscribeSpeechService

from .audio_fetcher.models import PlaylistInfo
from .models import ActorTranscript

logger = logging.getLogger(__name__)


class SpeechToTextService:
    """Umbrella speech2text: URL/video → ActorTranscript turn-level.

    No incluye segmentación ni temas.
    """

    def __init__(
        self,
        data_dir: Path = Path("data"),
        actor: str = "Lilly Téllez",
        video_ref_id: str = "su9nURIj9XQ",
        whisper_model: str = "large-v3-turbo",
        idioma: str = "es",
        umbral_match: float = 0.5,
        umbral_ambiguo: float = 0.7,
        pipeline_name: str = "pyannote/speaker-diarization-community-1",
        fallback_pipeline: str | None = "pyannote-community/speaker-diarization-community-1",
        device: str = "auto",
        audio_fetcher: AudioFetcherService | None = None,
        speaker_timestamps: SpeakerTimestampsService | None = None,
        transcribe_speech: TranscribeSpeechService | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.actor = actor
        self.video_ref_id = video_ref_id
        self.audio_fetcher = audio_fetcher or AudioFetcherService(data_dir=data_dir)
        self.speaker_timestamps = speaker_timestamps or SpeakerTimestampsService(
            actor=actor,
            video_ref_id=video_ref_id,
            pipeline_name=pipeline_name,
            fallback_pipeline=fallback_pipeline,
            device=device,
            umbral_match=umbral_match,
            umbral_ambiguo=umbral_ambiguo,
        )
        self.transcribe_speech = transcribe_speech or TranscribeSpeechService(
            actor=actor,
            whisper_model=whisper_model,
            idioma=idioma,
        )
        self._perfil_ready = False

    def discover(self, url_playlist: str) -> tuple[PlaylistInfo, list[VideoMeta]]:
        """Mapa de la playlist (sin descargar audio)."""
        return self.audio_fetcher.discover(url_playlist)

    def ensure_perfil(self, nombre_playlist: str, metas: list[VideoMeta]) -> bool:
        """Descarga el video de referencia y construye el perfil de voz.

        Returns:
            True si el perfil quedó listo; False si no se pudo obtener el audio ref.
        """
        if self._perfil_ready:
            return True

        ref_meta = next((m for m in metas if m.video_id == self.video_ref_id), None)
        if ref_meta is None:
            logger.error(
                "video_ref_id=%r no está en la playlist; no se puede construir perfil",
                self.video_ref_id,
            )
            return False

        ref_audio = self.audio_fetcher.fetch_one(ref_meta, nombre_playlist)
        if ref_audio is None:
            logger.error("No se pudo descargar audio de referencia %s", self.video_ref_id)
            return False

        self.speaker_timestamps.build_perfil(ref_audio)
        self._perfil_ready = True
        return True

    def procesar_one(
        self,
        video: VideoMeta,
        nombre_playlist: str,
        *,
        archive_path: Path | None = None,
    ) -> ActorTranscript | None:
        """Unidad del loop Fase 1: fetch → speakers → ASR actor.

        Returns:
            ActorTranscript o None (skip: fallo descarga / sin actor / sin texto).
        """
        audio = self.audio_fetcher.fetch_one(
            video,
            nombre_playlist,
            archive_path=archive_path,
        )
        if audio is None:
            return None

        turnos = self.speaker_timestamps.procesar_one(audio)
        if not turnos:
            return None

        return self.transcribe_speech.procesar_one(audio, turnos)

    def procesar(self, url_playlist: str) -> list[ActorTranscript]:
        """Wrapper ad-hoc: discover + perfil + procesar_one×N.

        No es el camino del PipelineRunner (que también segmenta y corre Temas).
        """
        playlist, metas = self.discover(url_playlist)
        if not metas:
            return []

        if not self.ensure_perfil(playlist.nombre, metas):
            return []

        resultados: list[ActorTranscript] = []
        for meta in metas:
            tx = self.procesar_one(meta, playlist.nombre)
            if tx is not None:
                resultados.append(tx)
        return resultados

    def fetch_one(
        self,
        video: VideoMeta,
        nombre_playlist: str,
        *,
        archive_path: Path | None = None,
    ) -> AudioVideo | None:
        """Atajo a audio_fetcher.fetch_one (p.ej. cleanup de .wav en el runner)."""
        return self.audio_fetcher.fetch_one(
            video,
            nombre_playlist,
            archive_path=archive_path,
        )
