"""Obtención de metadatos de playlists de YouTube."""

from __future__ import annotations

import json
import logging
import re
import subprocess

from ..models import PlaylistInfo, VideoInfo

logger = logging.getLogger(__name__)


def obtener_info_playlist(url: str) -> PlaylistInfo:
    """Obtiene los metadatos de una playlist de YouTube sin descargar audio.

    Usa yt-dlp en modo --flat-playlist para extraer rápidamente:
    - Nombre de la playlist
    - Lista de videos con id, título, url, duración y fecha aproximada

    Los videos privados, eliminados o inaccesibles se filtran automáticamente:
    yt-dlp los omite de la salida --flat-playlist.

    Args:
        url: URL completa de la playlist de YouTube.

    Returns:
        PlaylistInfo con el nombre y la lista de videos.

    Raises:
        RuntimeError: Si yt-dlp falla o la playlist no es accesible.
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--extractor-args", "youtubetab:approximate_date",
        "-j",
        "--no-warnings",
        url,
    ]

    resultado = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120
    )

    if resultado.returncode != 0:
        raise RuntimeError(
            f"yt-dlp falló al obtener info de la playlist: {resultado.stderr.strip()}"
        )

    nombre = "playlist_sin_nombre"
    videos: list[VideoInfo] = []

    for linea in resultado.stdout.strip().splitlines():
        linea = linea.strip()
        if not linea or not linea.startswith("{"):
            continue
        try:
            data = json.loads(linea)
        except json.JSONDecodeError:
            logger.warning(f"No se pudo parsear línea JSON: {linea[:80]}...")
            continue

        if nombre == "playlist_sin_nombre":
            nombre = data.get("playlist") or "playlist_sin_nombre"

        video_id = data.get("id")
        if not video_id:
            continue

        upload_date = data.get("upload_date")
        fecha = upload_date if upload_date and upload_date != "NA" else None

        videos.append(
            VideoInfo(
                id=video_id,
                titulo=data.get("title", "Sin título"),
                url=data.get("url", f"https://www.youtube.com/watch?v={video_id}"),
                duracion=data.get("duration") or 0.0,
                fecha=fecha,
            )
        )

    nombre = sanitizar_nombre_directorio(nombre)

    logger.info(f"Playlist '{nombre}': {len(videos)} videos encontrados")

    return PlaylistInfo(nombre=nombre, url=url, videos=videos)


def sanitizar_nombre_directorio(nombre: str) -> str:
    """Sanitiza un string para usarlo como nombre de directorio seguro."""
    nombre = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", nombre)
    nombre = re.sub(r"\s+", "_", nombre)
    nombre = nombre.strip("_.")
    return nombre or "playlist_sin_nombre"
