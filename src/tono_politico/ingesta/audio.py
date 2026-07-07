"""Descarga y cache de audios .wav desde YouTube."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from ..models import VideoInfo
from .cache import ruta_audio, ruta_dir_videos

logger = logging.getLogger(__name__)


def verificar_cache_videos(
    nombre_playlist: str,
    videos: list[VideoInfo],
    base_dir: Path | None = None,
) -> dict[str, list[VideoInfo]]:
    """Verifica qué audios ya están descargados en cache.

    Revisa si existe el directorio videos-<nombre_playlist>/ y compara
    los archivos .wav contra los video_ids de la playlist.

    Args:
        nombre_playlist: Nombre sanitizado de la playlist.
        videos: Lista de videos de la playlist.
        base_dir: Directorio raíz de datos (default: DATA_DIR).

    Returns:
        Dict con "existentes" y "faltantes".
    """
    dir_videos = ruta_dir_videos(nombre_playlist, base_dir)

    if not dir_videos.exists():
        logger.info(f"Cache de videos no existe: {dir_videos}")
        return {"existentes": [], "faltantes": videos}

    existentes: list[VideoInfo] = []
    faltantes: list[VideoInfo] = []

    for video in videos:
        if ruta_audio(nombre_playlist, video.id, base_dir).exists():
            existentes.append(video)
        else:
            faltantes.append(video)

    logger.info(
        f"Cache videos: {len(existentes)} existentes, {len(faltantes)} faltantes"
    )

    return {"existentes": existentes, "faltantes": faltantes}


def descargar_audio(
    video: VideoInfo,
    nombre_playlist: str,
    base_dir: Path | None = None,
) -> Path | None:
    """Descarga solo el audio de un video de YouTube como WAV.

    Usa yt-dlp para extraer el audio en formato WAV (compatible con Whisper).
    El archivo se guarda como {video_id}.wav en el directorio de cache.

    Si la descarga falla (HTTP 403, video privado, etc.), registra el error
    y devuelve None en vez de crashear — el pipeline salta el video y continúa.

    Args:
        video: VideoInfo del video a descargar.
        nombre_playlist: Nombre sanitizado de la playlist.
        base_dir: Directorio raíz de datos (default: DATA_DIR).

    Returns:
        Path al archivo de audio descargado, o None si falló.
    """
    dir_videos = ruta_dir_videos(nombre_playlist, base_dir)
    dir_videos.mkdir(parents=True, exist_ok=True)

    destino = ruta_audio(nombre_playlist, video.id, base_dir)
    url = f"https://www.youtube.com/watch?v={video.id}"

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "-f", "bestaudio/best",
        "-o", str(destino),
        "--no-warnings",
        "--retries", "10",
        url,
    ]

    logger.info(f"Descargando audio: [{video.id}] {video.titulo[:60]}")

    try:
        resultado = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout descargando video {video.id}, saltando")
        return None

    if resultado.returncode != 0:
        logger.error(
            f"Error descargando video {video.id}: "
            f"{resultado.stderr.strip()[:200]}, saltando"
        )
        return None

    if not destino.exists():
        logger.error(
            f"Descarga de {video.id} completada pero el archivo no existe, saltando"
        )
        return None

    tamanio_mb = destino.stat().st_size / (1024 * 1024)
    logger.info(f"Audio descargado: {destino.name} ({tamanio_mb:.1f} MB)")

    return destino
