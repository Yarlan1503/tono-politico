"""AudioFetcherService — discover + fetch_one (sin Whisper)."""

from __future__ import annotations

import logging
from pathlib import Path

from tono_politico.models import PlaylistInfo

from .audio import descargar_audio_result, verificar_cache_videos
from .cache import ruta_audio
from .models import AudioVideo, VideoMeta
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
        if destino.exists():
            logger.info(f"Audio en cache: {destino.name}")
            return AudioVideo.from_meta(video, audio_path=destino)

        result = descargar_audio_result(
            video,
            nombre_playlist,
            self.data_dir,
            archive_path=archive_path,
        )
        if not result.ok or result.path is None:
            logger.warning(
                "fetch_one skip [%s]: %s",
                video.video_id,
                result.error or "sin path",
            )
            return None
        return AudioVideo.from_meta(video, audio_path=result.path)

    def procesar(self, url_playlist: str) -> list[AudioVideo]:
        """Wrapper ad-hoc: discover + fetch_one×N.

        No es el camino del PipelineRunner en producción (loop por video
        con diarización/ASR). Útil en tests y uso suelto.
        """
        playlist, metas = self.discover(url_playlist)
        if not metas:
            logger.info("Playlist vacía, no hay nada que descargar")
            return []

        estado = verificar_cache_videos(playlist.nombre, metas, self.data_dir)
        logger.info(
            "Audios: %s en cache, %s por descargar",
            len(estado["existentes"]),
            len(estado["faltantes"]),
        )

        resultados: list[AudioVideo] = []
        for meta in metas:
            audio = self.fetch_one(meta, playlist.nombre)
            if audio is not None:
                resultados.append(audio)
        return resultados
