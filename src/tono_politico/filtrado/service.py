"""Componente 4: Filtrado — service OOP."""

from __future__ import annotations

from ..temas.models import ResultadoTemas
from .filtro import filtrar_por_topico
from .models import CriterioFiltrado, ResultadoFiltrado


class FiltradoService:
    """Service del Componente 4: filtrado por tópico seleccionado."""

    def __init__(
        self,
        topico_id: int,
        min_relevancia: float = 0.35,
        incluir_outliers: bool = False,
    ) -> None:
        self.topico_id = topico_id
        self.min_relevancia = min_relevancia
        self.incluir_outliers = incluir_outliers

    def procesar(self, resultado_temas: ResultadoTemas) -> ResultadoFiltrado:
        """Filtra segmentos usando la configuración del constructor."""
        criterio = CriterioFiltrado(
            topico_id=self.topico_id,
            min_relevancia=self.min_relevancia,
            incluir_outliers=self.incluir_outliers,
        )
        return filtrar_por_topico(resultado_temas, criterio)
