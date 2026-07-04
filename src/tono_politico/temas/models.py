"""DTOs del Componente 3: Temas."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..segmentacion.models import Segmento


@dataclass
class SegmentoTematizado:
    """Segmento enriquecido con su asignación de tópico.

    Atributos:
        segmento: El Segmento original del Componente 2.
        topico_id: ID del tópico asignado por BERTopic (-1 = outlier).
        probabilidad: Confianza de la asignación (0.0 a 1.0).
    """

    segmento: Segmento
    topico_id: int
    probabilidad: float


@dataclass
class TopicoInfo:
    """Metadata de un tópico descubierto por BERTopic.

    Atributos:
        id: ID del tópico (-1 = outlier/ruido).
        nombre: Etiqueta auto-generada por BERTopic (top words).
        palabras_clave: Top-N términos representativos vía c-TF-IDF.
        num_segmentos: Cuántos segmentos pertenecen a este tópico.
        representatividad: Porcentaje del corpus total (0.0 a 1.0).
    """

    id: int
    nombre: str
    palabras_clave: list[str] = field(default_factory=list)
    num_segmentos: int = 0
    representatividad: float = 0.0


@dataclass
class ResultadoTemas:
    """Salida completa del Componente 3.

    Atributos:
        segmentos: Lista de SegmentoTematizado (segmentos con su tópico).
        topicos: Lista de TopicoInfo (metadata de cada tópico).
        num_topicos: Número de tópicos descubiertos (excluyendo outliers -1).
    """

    segmentos: list[SegmentoTematizado] = field(default_factory=list)
    topicos: list[TopicoInfo] = field(default_factory=list)
    num_topicos: int = 0
