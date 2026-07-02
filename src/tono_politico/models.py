"""Modelos de datos compartidos entre componentes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideoInfo:
    """Metadatos básicos de un video de la playlist."""

    id: str
    titulo: str
    url: str
    duracion: float  # segundos
    fecha: Optional[str] = None  # YYYYMMDD — se llena al descargar audio


@dataclass
class SegmentoRaw:
    """Segmento crudo producido por Whisper (ventana acústica)."""

    texto: str
    t_start: float
    t_end: float
    pausa_antes: float  # gap entre este segmento y el anterior


@dataclass
class VideoTranscript:
    """Transcripción completa de un video — salida del Componente 1."""

    video_id: str
    url: str
    titulo: str
    fecha: Optional[str]
    raw_segments: list[SegmentoRaw] = field(default_factory=list)


@dataclass
class PlaylistInfo:
    """Información de una playlist de YouTube."""

    nombre: str
    url: str
    videos: list[VideoInfo] = field(default_factory=list)
