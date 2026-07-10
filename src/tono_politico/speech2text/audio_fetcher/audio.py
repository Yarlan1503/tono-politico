"""Descarga y verificación de cache de audios .wav desde YouTube."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from .models import DownloadResult, VideoMeta

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


def ruta_dir_videos(nombre_playlist: str, base_dir: Path | None = None) -> Path:
    """Devuelve la ruta al directorio de audios de una playlist."""
    base = base_dir or DATA_DIR
    return base / nombre_playlist / f"videos-{nombre_playlist}"


def ruta_audio(nombre_playlist: str, video_id: str, base_dir: Path | None = None) -> Path:
    """Devuelve la ruta al archivo WAV de un vídeo."""
    return ruta_dir_videos(nombre_playlist, base_dir) / f"{video_id}.wav"


def _audio_cache_valido(path: Path) -> bool:
    """Comprueba que el cache sea un archivo regular no vacío."""
    return path.is_file() and path.stat().st_size > 0


def verificar_cache_videos(
    nombre_playlist: str,
    videos: list[VideoMeta],
    base_dir: Path | None = None,
) -> dict[str, list[VideoMeta]]:
    """Verifica qué audios ya están descargados en cache.

    Returns:
        Dict con ``existentes`` y ``faltantes`` (listas de ``VideoMeta``).
    """
    dir_videos = ruta_dir_videos(nombre_playlist, base_dir)

    if not dir_videos.exists():
        logger.info(f"Cache de videos no existe: {dir_videos}")
        return {"existentes": [], "faltantes": list(videos)}

    existentes: list[VideoMeta] = []
    faltantes: list[VideoMeta] = []

    for video in videos:
        if _audio_cache_valido(ruta_audio(nombre_playlist, video.video_id, base_dir)):
            existentes.append(video)
        else:
            faltantes.append(video)

    logger.info(f"Cache videos: {len(existentes)} existentes, {len(faltantes)} faltantes")
    return {"existentes": existentes, "faltantes": faltantes}


def descargar_audio_result(
    video: VideoMeta,
    nombre_playlist: str,
    base_dir: Path | None = None,
    archive_path: Path | None = None,
) -> DownloadResult:
    """Descarga solo el audio de un video de YouTube como WAV.

    Devuelve un ``DownloadResult`` estructurado. No crashea: los fallos
    quedan en ``ok=False`` + ``error``.
    """
    dir_videos = ruta_dir_videos(nombre_playlist, base_dir)
    dir_videos.mkdir(parents=True, exist_ok=True)

    destino = ruta_audio(nombre_playlist, video.video_id, base_dir)
    url = f"https://www.youtube.com/watch?v={video.video_id}"

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format",
        "wav",
        "-f",
        "bestaudio/best",
        "-o",
        str(destino),
        "--no-warnings",
        "--retries",
        "10",
    ]

    if archive_path is not None:
        cmd.extend(["--download-archive", str(archive_path)])
        logger.debug("Usando download archive: %s", archive_path)

    cmd.append(url)
    logger.info(f"Descargando audio: [{video.video_id}] {video.titulo[:60]}")

    try:
        resultado = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        error = "yt-dlp no está instalado o no está en PATH"
        logger.error(error)
        return DownloadResult(video_id=video.video_id, path=None, ok=False, error=error)
    except subprocess.TimeoutExpired:
        error = f"Timeout después de 600s descargando video {video.video_id}"
        logger.error(error)
        return DownloadResult(video_id=video.video_id, path=None, ok=False, error=error)

    if resultado.returncode != 0:
        error = resultado.stderr.strip()[:300] or f"yt-dlp exit code {resultado.returncode}"
        logger.error(f"Error descargando video {video.video_id}: {error[:200]}, saltando")
        return DownloadResult(video_id=video.video_id, path=None, ok=False, error=error)

    if not _audio_cache_valido(destino):
        error = f"Descarga de {video.video_id} completada pero el audio no es un archivo válido"
        logger.error(error)
        return DownloadResult(video_id=video.video_id, path=None, ok=False, error=error)

    tamanio_mb = destino.stat().st_size / (1024 * 1024)
    logger.info(f"Audio descargado: {destino.name} ({tamanio_mb:.1f} MB)")
    return DownloadResult(video_id=video.video_id, path=destino, ok=True)
