"""DTOs del Componente 4: Filtrado."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..segmentacion.models import Segmento
from ..temas.models import TopicoInfo


@dataclass
class CriterioFiltrado:
    """Criterio para seleccionar segmentos de un tópico específico.

    Atributos:
        topico_id: ID del tópico elegido en el Componente 3.
        min_relevancia: Probabilidad mínima de asignación para incluir un segmento.
        incluir_outliers: Permite filtrar explícitamente el tópico -1 si se solicita.
    """

    topico_id: int
    min_relevancia: float = 0.35
    incluir_outliers: bool = False


@dataclass
class SegmentoFiltrado:
    """Segmento que superó el criterio de filtrado.

    Atributos:
        segmento: Segmento original del Componente 2.
        topico_id: Tópico por el que fue seleccionado.
        relevancia: Probabilidad/relevancia usada para el filtro.
    """

    segmento: Segmento
    topico_id: int
    relevancia: float


@dataclass
class ResultadoFiltrado:
    """Salida del Componente 4.

    Atributos:
        criterio: Criterio usado para filtrar.
        topico: Metadata del tópico elegido, si existe en ResultadoTemas.
        segmentos: Segmentos que cumplen el criterio.
        total_segmentos_entrada: Total de segmentos tematizados recibidos.
        total_segmentos_filtrados: Total de segmentos seleccionados.
    """

    criterio: CriterioFiltrado
    topico: TopicoInfo | None
    segmentos: list[SegmentoFiltrado] = field(default_factory=list)
    total_segmentos_entrada: int = 0
    total_segmentos_filtrados: int = 0
