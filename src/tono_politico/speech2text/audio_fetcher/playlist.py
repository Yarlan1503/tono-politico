"""Obtención de metadatos de playlists de YouTube (sin descarga de audio)."""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from datetime import datetime

from .models import PlaylistInfo, VideoMeta

logger = logging.getLogger(__name__)


def sanitizar_nombre_directorio(nombre: str) -> str:
    """Sanitiza un string para usarlo como nombre de directorio seguro."""
    nombre = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", nombre)
    nombre = re.sub(r"\s+", "_", nombre)
    nombre = nombre.strip("_.")
    return nombre or "playlist_sin_nombre"


def _normalizar_fecha(data: dict[str, object]) -> str | None:
    """Obtiene una fecha válida priorizando upload_date sobre release_date."""
    for field in ("upload_date", "release_date"):
        value = data.get(field)
        if not isinstance(value, str) or value in {"", "NA"}:
            continue
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError:
            logger.warning("Fecha inválida en %s: %r", field, value)
            continue
        return value
    return None


def _normalizar_duracion(value: object) -> float:
    """Normaliza duración de yt-dlp; valores inválidos se tratan como ausentes."""
    if value is None:
        return 0.0
    if not isinstance(value, (int, float, str)):
        logger.warning("Duración inválida: %r", value)
        return 0.0
    try:
        duracion = float(value)
    except (TypeError, ValueError):
        logger.warning("Duración inválida: %r", value)
        return 0.0
    if duracion < 0 or not math.isfinite(duracion):
        logger.warning("Duración inválida: %r", value)
        return 0.0
    return duracion


def obtener_info_playlist(url: str) -> tuple[PlaylistInfo, list[VideoMeta]]:
    """Obtiene metadatos de una playlist sin descargar audio.

    Returns:
        Tupla ``(PlaylistInfo, list[VideoMeta])``.

        - ``PlaylistInfo.nombre``: nombre sanitizado de la playlist.
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

    try:
        resultado = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError as exc:
        raise RuntimeError("yt-dlp no está instalado o no está en PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("yt-dlp timeout al obtener la información de la playlist") from exc

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

        fecha = _normalizar_fecha(data)
        duracion = _normalizar_duracion(data.get("duration"))

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

    playlist = PlaylistInfo(nombre=nombre)
    return playlist, videos
