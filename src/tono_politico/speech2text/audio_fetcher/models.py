"""DTOs del subpaquete audio_fetcher (descarga de audio + metadata)."""

from __future__ import annotations

from dataclasses import dataclass, field
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

    @classmethod
    def from_meta(cls, meta: VideoMeta, *, audio_path: Path) -> AudioVideo:
        """Construye un ``AudioVideo`` a partir de metadata pre-descarga y la ruta del wav."""
        return cls(
            video_id=meta.video_id,
            url=meta.url,
            titulo=meta.titulo,
            fecha=meta.fecha,
            audio_path=audio_path,
            duracion=meta.duracion,
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
class VideoInfo:
    """Metadatos básicos de un video de la playlist."""

    id: str
    titulo: str
    url: str
    duracion: float  # segundos
    fecha: str | None = None  # YYYYMMDD


@dataclass
class PlaylistInfo:
    """Información de una playlist de YouTube."""

    nombre: str
    url: str
    videos: list[VideoInfo] = field(default_factory=list)
