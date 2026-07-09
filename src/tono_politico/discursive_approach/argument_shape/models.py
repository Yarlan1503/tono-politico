"""DTOs de argument_shape (ex-Segmento → Argumento)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Oracion:
    """Oración sin word-level obligatorio (path actor-only)."""

    texto: str
    t_start: float
    t_end: float


@dataclass
class Argumento:
    """Bloque semántico coherente dentro de un audio (ex-Segmento)."""

    texto: str
    t_start: float
    t_end: float
    oraciones: list[Oracion] = field(default_factory=list)
    word_count: int = 0
    video_id: str = ""
    fecha: str | None = None  # YYYYMMDD desde VideoMeta / ActorTranscript
