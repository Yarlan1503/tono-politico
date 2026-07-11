"""Obtención de metadatos de playlists de YouTube (sin descarga de audio)."""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

from .models import PlaylistInfo, VideoMeta

logger = logging.getLogger(__name__)


def sanitizar_nombre_directorio(nombre: str) -> str:
    """Sanitiza un string para usarlo como nombre de directorio seguro."""
    nombre = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", nombre)
    nombre = re.sub(r"\s+", "_", nombre)
    nombre = nombre.strip("_.")
    return nombre or "playlist_sin_nombre"


def _normalizar_fecha(data: dict[str, object]) -> tuple[str | None, str]:
    """Obtiene fecha y fuente, priorizando upload_date sobre release_date."""
    invalid = False
    for field in ("upload_date", "release_date"):
        value = data.get(field)
        if not isinstance(value, str) or value in {"", "NA"}:
            continue
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError:
            logger.warning("Fecha inválida en %s: %r", field, value)
            invalid = True
            continue
        return value, field

    timestamp = data.get("timestamp")
    if timestamp not in (None, "", "NA"):
        try:
            if isinstance(timestamp, str):
                if timestamp.replace(".", "", 1).isdigit():
                    parsed = datetime.fromtimestamp(float(timestamp), tz=UTC)
                else:
                    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=UTC)
            elif isinstance(timestamp, (int, float)):
                parsed = datetime.fromtimestamp(timestamp, tz=UTC)
            else:
                raise TypeError("timestamp no es numérico ni ISO")
            return parsed.strftime("%Y%m%d"), "timestamp"
        except (TypeError, ValueError, OverflowError):
            logger.warning("Fecha inválida en timestamp: %r", timestamp)
            invalid = True

    return None, "invalid" if invalid else "missing"


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

    nombre_original: str | None = None
    playlist_id: str | None = None
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

        if nombre_original is None:
            nombre_original = data.get("playlist_title") or data.get("playlist")
            playlist_id = data.get("playlist_id")

        video_id = data.get("id")
        if not video_id:
            continue

        fecha, fecha_fuente = _normalizar_fecha(data)
        duracion = _normalizar_duracion(data.get("duration"))

        videos.append(
            VideoMeta(
                video_id=video_id,
                url=data.get("url") or f"https://www.youtube.com/watch?v={video_id}",
                titulo=data.get("title") or "Sin título",
                fecha=fecha,
                duracion=duracion,
                fecha_fuente=fecha_fuente,
            )
        )

    nombre_original = nombre_original or "playlist_sin_nombre"
    nombre_cache = sanitizar_nombre_directorio(nombre_original)
    if playlist_id is None:
        playlist_id = parse_qs(urlparse(url).query).get("list", [None])[0]
    logger.info("Playlist '%s': %s videos encontrados", nombre_original, len(videos))

    playlist = PlaylistInfo(
        nombre=nombre_original,
        nombre_cache=nombre_cache,
        playlist_id=playlist_id,
        url=url,
    )
    return playlist, videos
