"""Componente 4: Filtrado.

API pública:
    FiltradoService — service OOP para filtrar por tópico seleccionado.
    filtrar_por_topico — función pura para seleccionar segmentos relevantes.
    CriterioFiltrado, SegmentoFiltrado, ResultadoFiltrado — DTOs del componente.
"""

from .filtro import filtrar_por_topico
from .models import CriterioFiltrado, ResultadoFiltrado, SegmentoFiltrado
from .service import FiltradoService

__all__ = [
    "CriterioFiltrado",
    "FiltradoService",
    "ResultadoFiltrado",
    "SegmentoFiltrado",
    "filtrar_por_topico",
]
