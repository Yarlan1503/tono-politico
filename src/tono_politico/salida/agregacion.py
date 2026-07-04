"""Funciones puras de agregación.

Colapsan un ResultadoTono (N segmentos con scores por dimensión) en un
PerfilActor interpretable: label dominante por dimensión, intensidad
promedio y stance mayoritario.
"""

from __future__ import annotations

from ..tono.models import ResultadoTono, SegmentoConTono
from .models import PerfilActor


def stance_dominante(segmentos: list[SegmentoConTono]) -> str:
    """Devuelve el stance mayoritario entre todos los segmentos.

    En caso de empate, devuelve "apoyo" (default conservador).

    Args:
        segmentos: Lista de segmentos con tono analizado.

    Returns:
        "apoyo" o "rechazo".
    """
    if not segmentos:
        return "apoyo"

    apoyo = sum(1 for s in segmentos if s.stance.stance == "apoyo")
    rechazo = sum(1 for s in segmentos if s.stance.stance == "rechazo")

    if rechazo > apoyo:
        return "rechazo"
    return "apoyo"


def intensidad_promedio(segmentos: list[SegmentoConTono]) -> float:
    """Promedia la intensidad antagónica de todos los segmentos.

    Args:
        segmentos: Lista de segmentos con tono analizado.

    Returns:
        Promedio en escala 1-5. 0.0 si no hay segmentos.
    """
    if not segmentos:
        return 0.0
    return sum(s.intensidad_antagonica for s in segmentos) / len(segmentos)


def dimension_dominante(
    segmentos: list[SegmentoConTono],
    dimension: str,
) -> str:
    """Devuelve el label con mayor score promedio en una dimensión.

    Args:
        segmentos: Lista de segmentos con tono analizado.
        dimension: Nombre del atributo del SegmentoConTono
            ("logica_politica", "sentimiento", "estilo_discursivo",
            "funcion_discursiva").

    Returns:
        Nombre del label dominante.

    Raises:
        ValueError: Si no hay segmentos.
    """
    if not segmentos:
        raise ValueError(
            "No se puede calcular dimensión dominante sin segmentos"
        )

    # Obtener el DTO de la dimensión del primer segmento
    dto = getattr(segmentos[0], dimension)

    # Cada DTO tiene .to_scores() → list[EtiquetaScore]
    # Promediar cada label
    etiquetas = [e.etiqueta for e in dto.to_scores()]
    sumas = {et: 0.0 for et in etiquetas}

    for seg in segmentos:
        dto_seg = getattr(seg, dimension)
        for e in dto_seg.to_scores():
            sumas[e.etiqueta] += e.score

    promedios = {et: sumas[et] / len(segmentos) for et in etiquetas}

    return max(promedios, key=lambda k: promedios[k])


def generar_perfil(resultado_tono: ResultadoTono) -> PerfilActor:
    """Genera el perfil agregado completo del actor.

    Args:
        resultado_tono: Salida del Componente 5.

    Returns:
        PerfilActor con todas las dimensiones agregadas.
    """
    segmentos = resultado_tono.segmentos
    n = len(segmentos)

    if n == 0:
        return PerfilActor(
            actor=resultado_tono.actor,
            tema=resultado_tono.tema,
            n_segmentos=0,
            stance_dominante="apoyo",
            intensidad_promedio=0.0,
            logica_dominante="populista",
            sentimiento_dominante="esperanza",
            estilo_dominante="directo",
            funcion_dominante="propuesta",
        )

    return PerfilActor(
        actor=resultado_tono.actor,
        tema=resultado_tono.tema,
        n_segmentos=n,
        stance_dominante=stance_dominante(segmentos),
        intensidad_promedio=intensidad_promedio(segmentos),
        logica_dominante=dimension_dominante(segmentos, "logica_politica"),
        sentimiento_dominante=dimension_dominante(segmentos, "sentimiento"),
        estilo_dominante=dimension_dominante(segmentos, "estilo_discursivo"),
        funcion_dominante=dimension_dominante(segmentos, "funcion_discursiva"),
    )
