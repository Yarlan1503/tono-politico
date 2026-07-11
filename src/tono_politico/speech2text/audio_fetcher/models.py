"""DTOs del subpaquete audio_fetcher (descarga de audio + metadata)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoMeta:
    """Metadata de un video de playlist, previa a la descarga del audio.

    Attributes:
        video_id: ID del video de YouTube.
        url: URL del video.
        titulo: Título del video.
        fecha: Fecha de publicación YYYYMMDD, si está disponible.
        duracion: Duración en segundos (metadata de yt-dlp).
    """

    video_id: str
    url: str
    titulo: str
    fecha: str | None
    duracion: float
    fecha_fuente: str | None = None


@dataclass(frozen=True)
class AudioVideo:
    """Video con audio local listo para diarización / ASR.

    Attributes:
        video_id: ID del video de YouTube.
        url: URL del video.
        titulo: Título del video.
        fecha: Fecha de publicación YYYYMMDD, si está disponible.
        audio_path: Ruta al ``.wav`` descargado y cacheado.
        duracion: Duración en segundos (metadata de yt-dlp).
    """

    video_id: str
    url: str
    titulo: str
    fecha: str | None
    audio_path: Path
    duracion: float
    fecha_fuente: str | None = None
    playlist: PlaylistInfo | None = None

    @classmethod
    def from_meta(
        cls,
        meta: VideoMeta,
        *,
        audio_path: Path,
        playlist: PlaylistInfo | None = None,
    ) -> AudioVideo:
        """Construye un ``AudioVideo`` a partir de metadata pre-descarga y la ruta del wav."""
        return cls(
            video_id=meta.video_id,
            url=meta.url,
            titulo=meta.titulo,
            fecha=meta.fecha,
            audio_path=audio_path,
            duracion=meta.duracion,
            fecha_fuente=meta.fecha_fuente,
            playlist=playlist,
        )


@dataclass(frozen=True)
class DownloadResult:
    """Resultado estructurado de la descarga de un video.

    Attributes:
        video_id: ID del video de YouTube.
        path: Ruta al archivo de audio si la descarga fue exitosa, None si falló.
        ok: True si el archivo existe y es válido.
        error: Mensaje de error truncado si ok=False, None si ok=True.
    """

    video_id: str
    path: Path | None
    ok: bool
    error: str | None = None


@dataclass
class PlaylistInfo:
    """Identidad visible y de cache de una playlist."""

    nombre: str
    nombre_cache: str | None = None
    playlist_id: str | None = None
    url: str | None = None

    @property
    def cache_name(self) -> str:
        """Devuelve la clave de cache, con fallback para fixtures legacy."""
        return self.nombre_cache or self.nombre
