"""Adaptador Argumento ↔ API legacy de Tono (Segmento / ResultadoFiltrado)."""

from __future__ import annotations

from ...filtrado.models import CriterioFiltrado, ResultadoFiltrado, SegmentoFiltrado
from ...segmentacion.models import Oracion as OracionLegacy
from ...segmentacion.models import Segmento
from ...temas.models import TopicoInfo as TopicoInfoLegacy
from ...tono.models import ResultadoTono, SegmentoConTono
from ..argument_shape.models import Argumento
from ..topics_cluster.models import TopicoInfo
from .models import PerfilTonoArgumento


def argumento_a_segmento(argumento: Argumento) -> Segmento:
    """Convierte Argumento a Segmento legacy (sin words)."""
    oraciones = [
        OracionLegacy(texto=o.texto, t_start=o.t_start, t_end=o.t_end, words=[])
        for o in argumento.oraciones
    ]
    return Segmento(
        texto=argumento.texto,
        t_start=argumento.t_start,
        t_end=argumento.t_end,
        oraciones=oraciones,
        word_count=argumento.word_count,
        video_id=argumento.video_id,
    )


def topico_a_legacy(topico: TopicoInfo) -> TopicoInfoLegacy:
    """Mapea TopicoInfo de discursive_approach al DTO legacy de temas/filtrado."""
    return TopicoInfoLegacy(
        id=topico.id,
        nombre=topico.nombre,
        palabras_clave=list(topico.palabras_clave),
        num_segmentos=topico.num_argumentos,
        representatividad=topico.representatividad,
    )


def argumentos_a_resultado_filtrado(
    argumentos: list[Argumento],
    topico: TopicoInfo,
) -> ResultadoFiltrado:
    """Envuelve argumentos como ResultadoFiltrado para TonoService.procesar."""
    segmentos = [
        SegmentoFiltrado(
            segmento=argumento_a_segmento(a),
            topico_id=topico.id,
            relevancia=1.0,
        )
        for a in argumentos
    ]
    return ResultadoFiltrado(
        criterio=CriterioFiltrado(topico_id=topico.id, min_relevancia=0.0),
        topico=topico_a_legacy(topico),
        segmentos=segmentos,
        total_segmentos_entrada=len(argumentos),
        total_segmentos_filtrados=len(segmentos),
    )


def segmento_con_tono_a_perfil(sct: SegmentoConTono) -> PerfilTonoArgumento:
    return PerfilTonoArgumento(
        stance=sct.stance.stance,
        intensidad=sct.intensidad_antagonica,
        logica_dominante=sct.logica_politica.dominante().etiqueta,
        sentimiento_dominante=sct.sentimiento.dominante().etiqueta,
        estilo_dominante=sct.estilo_discursivo.dominante().etiqueta,
        funcion_dominante=sct.funcion_discursiva.dominante().etiqueta,
        raw=sct,
    )


def resultado_tono_a_perfiles(resultado: ResultadoTono) -> list[PerfilTonoArgumento]:
    return [segmento_con_tono_a_perfil(s) for s in resultado.segmentos]
