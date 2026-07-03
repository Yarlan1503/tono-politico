"""Modelos de datos compartidos entre componentes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WordTimestamp:
    """Timestamp de una palabra individual dentro de un segmento."""

    word: str
    start: float
    end: float
    probability: float | None = None


@dataclass
class SegmentoRaw:
    """Segmento crudo producido por Whisper (ventana acústica).

    Atributos:
        texto: Texto transcrito del segmento.
        t_start: Tiempo de inicio en segundos.
        t_end: Tiempo de fin en segundos.
        pausa_antes: Gap en segundos entre este segmento y el anterior
            (t_start[n] - t_end[n-1]). Para el primer segmento es 0.0.
        words: Timestamps por palabra producidos por Whisper.
    """

    texto: str
    t_start: float
    t_end: float
    pausa_antes: float
    words: list[WordTimestamp] = field(default_factory=list)


@dataclass
class VideoTranscript:
    """Transcripción completa de un video — salida del Componente 1.

    Atributos:
        video_id: ID del video de YouTube.
        url: URL del video.
        titulo: Título del video.
        fecha: Fecha de publicación en formato YYYYMMDD (o None).
        raw_segments: Lista de segmentos transcritos.
    """

    video_id: str
    url: str
    titulo: str
    fecha: str | None
    raw_segments: list[SegmentoRaw] = field(default_factory=list)


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
