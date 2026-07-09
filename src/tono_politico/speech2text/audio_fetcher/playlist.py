"""Obtención de metadatos de playlists de YouTube (sin descarga de audio)."""

from __future__ import annotations

import json
import logging
import re
import subprocess

from .models import PlaylistInfo, VideoMeta

logger = logging.getLogger(__name__)


def sanitizar_nombre_directorio(nombre: str) -> str:
    """Sanitiza un string para usarlo como nombre de directorio seguro."""
    nombre = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", nombre)
    nombre = re.sub(r"\s+", "_", nombre)
    nombre = nombre.strip("_.")
    return nombre or "playlist_sin_nombre"


def obtener_info_playlist(url: str) -> tuple[PlaylistInfo, list[VideoMeta]]:
    """Obtiene metadatos de una playlist sin descargar audio.

    Returns:
        Tupla ``(PlaylistInfo, list[VideoMeta])``.

        - ``PlaylistInfo.nombre``: nombre sanitizado de la playlist.
          (Campos legacy ``url`` / ``videos`` se rellenan de forma mínima
          hasta el slim-down completo de ``models.py`` compartido.)
        - ``list[VideoMeta]``: videos con id/url/titulo/fecha/duracion.

    Raises:
        RuntimeError: Si yt-dlp falla o la playlist no es accesible.
    """
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--extractor-args",
        "youtubetab:approximate_date",
        "-j",
        "--no-warnings",
        url,
    ]

    resultado = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if resultado.returncode != 0:
        raise RuntimeError(
            f"yt-dlp falló al obtener info de la playlist: {resultado.stderr.strip()}"
        )

    nombre = "playlist_sin_nombre"
    videos: list[VideoMeta] = []

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
        duration_raw = data.get("duration")
        duracion = float(duration_raw) if duration_raw is not None else 0.0

        videos.append(
            VideoMeta(
                video_id=video_id,
                url=data.get("url") or f"https://www.youtube.com/watch?v={video_id}",
                titulo=data.get("title") or "Sin título",
                fecha=fecha,
                duracion=duracion,
            )
        )

    nombre = sanitizar_nombre_directorio(nombre)
    logger.info(f"Playlist '{nombre}': {len(videos)} videos encontrados")

    # Hasta slim-down de PlaylistInfo: nombre es la fuente de verdad;
    # url se conserva por compatibilidad; videos vacío (viven en VideoMeta).
    playlist = PlaylistInfo(nombre=nombre, url=url, videos=[])
    return playlist, videos
