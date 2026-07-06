"""Transcripción con Whisper y persistencia de transcripciones JSON.

Tres responsabilidades:
    1. transcribir(): Ejecutar Whisper y normalizar la salida a SegmentoRaw.
    2. persistencia: guardar/cargar/verificar transcripciones en disco.
    3. (1) y (2) comparten el mismo formato JSON vía _serializar/_deserializar.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

from ..models import SegmentoRaw, VideoInfo, VideoTranscript, WordTimestamp
from .cache import ruta_dir_transcripciones, ruta_transcripcion

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Transcripción
# ──────────────────────────────────────────────────────────

def transcribir(
    audio_path: str | Path, modelo: str = "large-v3-turbo", idioma: str = "es"
) -> list[SegmentoRaw]:
    """Transcribe un archivo de audio con OpenAI Whisper oficial.

    Convierte la salida de Whisper a SegmentoRaw con pausa_antes calculada
    (gap acústico respecto al segmento anterior) y words como WordTimestamp.

    Args:
        audio_path: Ruta al archivo de audio local.
        modelo: Nombre del modelo Whisper a cargar (por defecto large-v3-turbo).
        idioma: Código de idioma para forzar la transcripción (por defecto es).

    Returns:
        Lista de SegmentoRaw con texto, timestamps, pausa_antes y words.

    Raises:
        FileNotFoundError: Si el archivo de audio no existe.
    """
    ruta_audio = Path(audio_path)
    if not ruta_audio.exists():
        raise FileNotFoundError(f"Audio no encontrado: {ruta_audio}")

    import whisper  # type: ignore[import-not-found]

    logger.info(f"Cargando modelo Whisper: {modelo}")
    modelo_whisper = whisper.load_model(modelo)

    logger.info(f"Transcribiendo audio: {ruta_audio}")
    resultado = modelo_whisper.transcribe(
        str(ruta_audio),
        language=idioma,
        word_timestamps=True,
        fp16=False,
    )

    segmentos: list[SegmentoRaw] = []
    t_end_anterior = 0.0

    for segmento in resultado.get("segments", []):
        texto = (segmento.get("text") or "").strip()
        if not texto:
            continue

        t_start = float(segmento.get("start", 0.0))
        t_end = float(segmento.get("end", 0.0))
        pausa = max(0.0, t_start - t_end_anterior) if segmentos else 0.0

        segmentos.append(
            SegmentoRaw(
                texto=texto,
                t_start=t_start,
                t_end=t_end,
                pausa_antes=pausa,
                words=_normalizar_words(segmento.get("words", [])),
            )
        )
        t_end_anterior = t_end

    logger.info(f"Transcripción generó {len(segmentos)} segmentos")
    return segmentos


def _normalizar_words(words: object) -> list[WordTimestamp]:
    """Convierte los timestamps por palabra de Whisper a WordTimestamp."""
    if not isinstance(words, list):
        return []

    normalizadas: list[WordTimestamp] = []
    for word_data in words:
        if not isinstance(word_data, dict):
            continue

        d = cast(dict[str, Any], word_data)
        palabra = str(d.get("word") or "").strip()
        if not palabra:
            continue

        probability = d.get("probability")
        normalizadas.append(
            WordTimestamp(
                word=palabra,
                start=float(d.get("start", 0.0)),
                end=float(d.get("end", 0.0)),
                probability=float(probability) if probability is not None else None,
            )
        )

    return normalizadas


# ──────────────────────────────────────────────────────────
# Persistencia
# ──────────────────────────────────────────────────────────

def verificar_cache_transcripciones(
    nombre_playlist: str,
    videos: list[VideoInfo],
    base_dir: Path | None = None,
) -> dict[str, list[VideoInfo]]:
    """Verifica qué transcripciones ya están en cache.

    Considera existente solo un JSON válido cuyo campo video_id coincida
    con el video esperado. Un JSON corrupto, vacío o con video_id distinto
    se trata como faltante.

    Args:
        nombre_playlist: Nombre sanitizado de la playlist.
        videos: Lista de videos de la playlist.
        base_dir: Directorio raíz de datos (default: DATA_DIR).

    Returns:
        Dict con "existentes" y "faltantes".
    """
    dir_t = ruta_dir_transcripciones(nombre_playlist, base_dir)

    if not dir_t.exists():
        logger.info(f"Cache de transcripciones no existe: {dir_t}")
        return {"existentes": [], "faltantes": videos}

    existentes: list[VideoInfo] = []
    faltantes: list[VideoInfo] = []

    for video in videos:
        ruta = ruta_transcripcion(nombre_playlist, video.id, base_dir)
        if _transcripcion_json_valida(ruta, video.id):
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


def guardar_transcripcion(
    transcript: VideoTranscript,
    nombre_playlist: str,
    base_dir: Path | None = None,
) -> Path:
    """Serializa un VideoTranscript a JSON en el directorio de cache.

    Args:
        transcript: VideoTranscript con los datos a persistir.
        nombre_playlist: Nombre sanitizado de la playlist.
        base_dir: Directorio raíz de datos (default: DATA_DIR).

    Returns:
        Path al archivo JSON escrito.
    """
    dir_t = ruta_dir_transcripciones(nombre_playlist, base_dir)
    dir_t.mkdir(parents=True, exist_ok=True)

    ruta_json = ruta_transcripcion(
        nombre_playlist, transcript.video_id, base_dir
    )

    ruta_json.write_text(
        json.dumps(_serializar_transcript(transcript), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        f"Transcripción guardada: {ruta_json.name} ({len(transcript.raw_segments)} segmentos)"
    )
    return ruta_json


def cargar_transcripcion(ruta: Path) -> VideoTranscript:
    """Reconstruye un VideoTranscript desde un JSON en disco.

    Inverso de guardar_transcripcion.

    Args:
        ruta: Path al archivo JSON.

    Returns:
        VideoTranscript reconstruido.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el JSON es inválido.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"Transcripción no encontrada: {ruta}")

    try:
        data = json.loads(ruta.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON inválido en {ruta}: {exc}") from exc

    return _deserializar_transcript(data)


# ──────────────────────────────────────────────────────────
# Serialización interna
# ──────────────────────────────────────────────────────────

def _serializar_transcript(transcript: VideoTranscript) -> dict:
    """Convierte un VideoTranscript a dict JSON-serializable."""
    return {
        "video_id": transcript.video_id,
        "url": transcript.url,
        "titulo": transcript.titulo,
        "fecha": transcript.fecha,
        "raw_segments": [
            {
                "texto": seg.texto,
                "t_start": seg.t_start,
                "t_end": seg.t_end,
                "pausa_antes": seg.pausa_antes,
                "words": [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                    for w in seg.words
                ],
            }
            for seg in transcript.raw_segments
        ],
    }


def _deserializar_transcript(data: dict) -> VideoTranscript:
    """Reconstruye un VideoTranscript desde un dict parsed de JSON."""
    raw_segments: list[SegmentoRaw] = []
    for seg_data in data.get("raw_segments", []):
        words = [
            WordTimestamp(
                word=str(w.get("word", "")),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                probability=w.get("probability"),
            )
            for w in seg_data.get("words", [])
        ]
        raw_segments.append(
            SegmentoRaw(
                texto=seg_data.get("texto", ""),
                t_start=float(seg_data.get("t_start", 0.0)),
                t_end=float(seg_data.get("t_end", 0.0)),
                pausa_antes=float(seg_data.get("pausa_antes", 0.0)),
                words=words,
            )
        )

    return VideoTranscript(
        video_id=data.get("video_id", ""),
        url=data.get("url", ""),
        titulo=data.get("titulo", ""),
        fecha=data.get("fecha"),
        raw_segments=raw_segments,
    )
