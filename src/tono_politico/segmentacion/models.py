"""DTOs del Componente 2: Segmentación."""

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
class Oracion:
    """Oración individual extraída de los segmentos crudos de Whisper.

    Atributos:
        texto: Texto de la oración.
        t_start: Tiempo de inicio en segundos (desde la primera word).
        t_end: Tiempo de fin en segundos (desde la última word).
        words: Timestamps por palabra que componen la oración.
    """

    texto: str
    t_start: float
    t_end: float
    words: list[WordTimestamp] = field(default_factory=list)


@dataclass
class Segmento:
    """Segmento semántico coherente — salida del Componente 2.

    Agrupa una o más oraciones que tratan un mismo tema, listas para
    análisis de tono.

    Atributos:
        texto: Texto completo del segmento (oraciones concatenadas).
        t_start: Tiempo de inicio del segmento.
        t_end: Tiempo de fin del segmento.
        oraciones: Oraciones que componen el segmento.
        word_count: Número total de palabras.
        video_id: ID del video de origen.
    """

    texto: str
    t_start: float
    t_end: float
    oraciones: list[Oracion] = field(default_factory=list)
    word_count: int = 0
    video_id: str = ""
