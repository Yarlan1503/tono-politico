"""Funciones puras de filtrado por tópico."""

from __future__ import annotations

from ..temas.models import ResultadoTemas, TopicoInfo
from .models import CriterioFiltrado, ResultadoFiltrado, SegmentoFiltrado


def filtrar_por_topico(
    resultado_temas: ResultadoTemas,
    criterio: CriterioFiltrado,
) -> ResultadoFiltrado:
    """Selecciona segmentos del tópico indicado que superan la relevancia mínima."""
    topico = _buscar_topico(resultado_temas.topicos, criterio.topico_id)
    segmentos: list[SegmentoFiltrado] = []

    for segmento_tematizado in resultado_temas.segmentos:
        if segmento_tematizado.topico_id != criterio.topico_id:
            continue
        if segmento_tematizado.topico_id == -1 and not criterio.incluir_outliers:
            continue
        if segmento_tematizado.probabilidad < criterio.min_relevancia:
            continue

        segmentos.append(
            SegmentoFiltrado(
                segmento=segmento_tematizado.segmento,
                topico_id=segmento_tematizado.topico_id,
                relevancia=segmento_tematizado.probabilidad,
            )
        )

    return ResultadoFiltrado(
        criterio=criterio,
        topico=topico,
        segmentos=segmentos,
        total_segmentos_entrada=len(resultado_temas.segmentos),
        total_segmentos_filtrados=len(segmentos),
    )


def _buscar_topico(topicos: list[TopicoInfo], topico_id: int) -> TopicoInfo | None:
    """Busca metadata de tópico por ID."""
    return next((topico for topico in topicos if topico.id == topico_id), None)
