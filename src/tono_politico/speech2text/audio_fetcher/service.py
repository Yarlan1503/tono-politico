"""AudioFetcherService — discover + fetch_one (sin Whisper)."""

from __future__ import annotations

import logging
from pathlib import Path

from .audio import _audio_cache_valido, descargar_audio_result, ruta_audio
from .models import AudioVideo, PlaylistInfo, VideoMeta
from .playlist import obtener_info_playlist

logger = logging.getLogger(__name__)


class AudioFetcherService:
    """Descarga de audio + metadata de playlists de YouTube.

    Attributes:
        data_dir: Directorio raíz del cache de audios.
    """

    def __init__(self, data_dir: Path = Path("data")) -> None:
        self.data_dir = data_dir

    def discover(self, url_playlist: str) -> tuple[PlaylistInfo, list[VideoMeta]]:
        """Mapa de la playlist sin descargar audio."""
        return obtener_info_playlist(url_playlist)

    def fetch_one(
        self,
        video: VideoMeta,
        nombre_playlist: str,
        *,
        archive_path: Path | None = None,
    ) -> AudioVideo | None:
        """Descarga (o reutiliza cache) el audio de un video.

        Returns:
            ``AudioVideo`` si el ``.wav`` está disponible; ``None`` si falla.
        """
        destino = ruta_audio(nombre_playlist, video.video_id, self.data_dir)
        if _audio_cache_valido(destino):
            logger.info(f"Audio en cache: {destino.name}")
            return AudioVideo.from_meta(video, audio_path=destino)

        result = descargar_audio_result(
            video,
            nombre_playlist,
            self.data_dir,
            archive_path=archive_path,
        )
        if not result.ok or result.path is None or not _audio_cache_valido(result.path):
            logger.warning(
                "fetch_one skip [%s]: %s",
                video.video_id,
                result.error or "sin path o audio inválido",
            )
            return None
        return AudioVideo.from_meta(video, audio_path=result.path)
