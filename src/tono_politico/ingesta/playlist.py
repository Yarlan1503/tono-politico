"""Componente 1: Ingesta.

Recibe la URL de una playlist de YouTube, descarga el audio de cada video,
lo transcribe con Whisper y devuelve transcripciones estructuradas con
timestamps y pausas entre segmentos.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

from ..models import PlaylistInfo, VideoInfo

logger = logging.getLogger(__name__)

# Directorio raíz para datos locales (cache de audios y transcripciones)
DATA_DIR = Path("data")


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
    # Una sola llamada con approximate_date para obtener todo: nombre, fecha, metadata
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--extractor-args", "youtubetab:approximate_date",
        "-j",  # JSON por línea, un objeto por video
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

    # Extraer nombre de la playlist del primer objeto JSON
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

        # El nombre viene en el primer objeto procesado
        if nombre == "playlist_sin_nombre":
            nombre = data.get("playlist") or "playlist_sin_nombre"

        video_id = data.get("id")
        if not video_id:
            continue

        # upload_date con approximate_date: formato YYYYMMDD o None
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

    # Sanitizar el nombre para usarlo como directorio
    nombre = _sanitizar_nombre_directorio(nombre)

    logger.info(
        f"Playlist '{nombre}': {len(videos)} videos encontrados"
    )

    return PlaylistInfo(nombre=nombre, url=url, videos=videos)


def _ruta_dir_videos(nombre_playlist: str) -> Path:
    """Devuelve la ruta al directorio de audios de una playlist."""
    return DATA_DIR / nombre_playlist / f"videos-{nombre_playlist}"


def verificar_cache_videos(
    nombre_playlist: str, videos: list[VideoInfo]
) -> dict[str, list[VideoInfo]]:
    """Verifica qué audios ya están descargados en cache.

    Revisa si existe el directorio videos-<nombre_playlist>/ y compara
    los archivos .wav contra los video_ids de la playlist.

    Args:
        nombre_playlist: Nombre sanitizado de la playlist.
        videos: Lista de videos de la playlist.

    Returns:
        Dict con dos listas:
        - "existentes": videos cuyo audio ya está en cache.
        - "faltantes": videos cuyo audio falta descargar.
    """
    dir_videos = _ruta_dir_videos(nombre_playlist)

    if not dir_videos.exists():
        logger.info(f"Cache de videos no existe: {dir_videos}")
        return {"existentes": [], "faltantes": videos}

    existentes: list[VideoInfo] = []
    faltantes: list[VideoInfo] = []

    for video in videos:
        ruta_audio = dir_videos / f"{video.id}.wav"
        if ruta_audio.exists():
            existentes.append(video)
        else:
            faltantes.append(video)

    logger.info(
        f"Cache videos: {len(existentes)} existentes, {len(faltantes)} faltantes"
    )

    return {"existentes": existentes, "faltantes": faltantes}


def _ruta_dir_transcripciones(nombre_playlist: str) -> Path:
    """Devuelve la ruta al directorio de transcripciones de una playlist."""
    return DATA_DIR / nombre_playlist / f"transcripciones-{nombre_playlist}"


def verificar_cache_transcripciones(
    nombre_playlist: str, videos: list[VideoInfo]
) -> dict[str, list[VideoInfo]]:
    """Verifica qué transcripciones ya están en cache.

    Revisa transcripciones-<nombre_playlist>/ y considera existente solo un
    JSON válido cuyo campo video_id coincida con el video esperado. Un JSON
    corrupto, vacío o con video_id distinto se trata como faltante para evitar
    reutilizar cachés rotos o equivocados.

    Args:
        nombre_playlist: Nombre sanitizado de la playlist.
        videos: Lista de videos de la playlist.

    Returns:
        Dict con dos listas:
        - "existentes": videos cuya transcripción JSON válida está en cache.
        - "faltantes": videos cuya transcripción falta o es inválida.
    """
    dir_transcripciones = _ruta_dir_transcripciones(nombre_playlist)

    if not dir_transcripciones.exists():
        logger.info(f"Cache de transcripciones no existe: {dir_transcripciones}")
        return {"existentes": [], "faltantes": videos}

    existentes: list[VideoInfo] = []
    faltantes: list[VideoInfo] = []

    for video in videos:
        ruta_transcripcion = dir_transcripciones / f"{video.id}.json"
        if _transcripcion_json_valida(ruta_transcripcion, video.id):
            existentes.append(video)
        else:
            faltantes.append(video)

    logger.info(
        f"Cache transcripciones: {len(existentes)} existentes, {len(faltantes)} faltantes"
    )

    return {"existentes": existentes, "faltantes": faltantes}


def _transcripcion_json_valida(ruta: Path, video_id: str) -> bool:
    """Indica si un JSON de transcripción existe, parsea y coincide con video_id."""
    if not ruta.exists():
        return False

    try:
        data = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning(f"Transcripción inválida o corrupta: {ruta}")
        return False

    return data.get("video_id") == video_id


def descargar_audio(
    video: VideoInfo, nombre_playlist: str
) -> Path:
    """Descarga solo el audio de un video de YouTube como WAV.

    Usa yt-dlp para extraer el audio en formato WAV (compatible con Whisper).
    El archivo se guarda como {video_id}.wav en el directorio de cache.

    Args:
        video: VideoInfo del video a descargar.
        nombre_playlist: Nombre sanitizado de la playlist.

    Returns:
        Path al archivo de audio descargado.

    Raises:
        RuntimeError: Si yt-dlp falla al descargar (video privado, eliminado, etc.).
    """
    dir_videos = _ruta_dir_videos(nombre_playlist)
    dir_videos.mkdir(parents=True, exist_ok=True)

    ruta_audio = dir_videos / f"{video.id}.wav"
    url = f"https://www.youtube.com/watch?v={video.id}"

    cmd = [
        "yt-dlp",
        "-x",                          # Extraer solo audio
        "--audio-format", "wav",       # Convertir a WAV
        "-f", "bestaudio/best",        # Mejor calidad de audio disponible
        "-o", str(ruta_audio),         # Ruta de salida exacta
        "--no-warnings",
        url,
    ]

    logger.info(f"Descargando audio: [{video.id}] {video.titulo[:60]}")

    resultado = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600
    )

    if resultado.returncode != 0:
        raise RuntimeError(
            f"Error descargando video {video.id}: {resultado.stderr.strip()}"
        )

    if not ruta_audio.exists():
        raise RuntimeError(
            f"Descarga completada pero el archivo no existe: {ruta_audio}"
        )

    tamanio_mb = ruta_audio.stat().st_size / (1024 * 1024)
    logger.info(f"Audio descargado: {ruta_audio.name} ({tamanio_mb:.1f} MB)")

    return ruta_audio


def transcribir(
    audio_path: str | Path, modelo: str = "large-v3", idioma: str = "es"
) -> list[dict[str, str | float]]:
    """Transcribe un archivo de audio con OpenAI Whisper oficial.

    Args:
        audio_path: Ruta al archivo de audio local.
        modelo: Nombre del modelo Whisper a cargar (por defecto large-v3).
        idioma: Código de idioma para forzar la transcripción (por defecto es).

    Returns:
        Lista de segmentos normalizados con texto y timestamps:
        [{"texto": str, "t_start": float, "t_end": float}, ...]

    Raises:
        FileNotFoundError: Si el archivo de audio no existe.
    """
    ruta_audio = Path(audio_path)
    if not ruta_audio.exists():
        raise FileNotFoundError(f"Audio no encontrado: {ruta_audio}")

    # Import diferido: permite testear sin tener Whisper instalado y evita coste al importar módulo.
    import whisper  # type: ignore[import-not-found]

    logger.info(f"Cargando modelo Whisper: {modelo}")
    modelo_whisper = whisper.load_model(modelo)

    logger.info(f"Transcribiendo audio: {ruta_audio}")
    resultado = modelo_whisper.transcribe(
        str(ruta_audio),
        language=idioma,
        word_timestamps=True,
        fp16=False,  # CPU-friendly; evita warning/error de fp16 sin GPU.
    )

    segmentos: list[dict[str, str | float]] = []
    for segmento in resultado.get("segments", []):
        texto = (segmento.get("text") or "").strip()
        if not texto:
            continue

        segmentos.append(
            {
                "texto": texto,
                "t_start": float(segmento.get("start", 0.0)),
                "t_end": float(segmento.get("end", 0.0)),
            }
        )

    logger.info(f"Transcripción generó {len(segmentos)} segmentos")
    return segmentos


def _sanitizar_nombre_directorio(nombre: str) -> str:
    """Sanitiza un string para usarlo como nombre de directorio seguro."""
    # Reemplazar caracteres problemáticos
    nombre = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", nombre)
    # Colapsar espacios múltiples
    nombre = re.sub(r"\s+", "_", nombre)
    # Eliminar guiones bajos al inicio/final
    nombre = nombre.strip("_.")
    return nombre or "playlist_sin_nombre"
