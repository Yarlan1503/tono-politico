"""Agrupación de oraciones en segmentos semánticos.

Toma la lista de Oracion + la lista de Breakpoint y produce list[Segmento].

Reglas:
    1. Los breakpoints dividen la secuencia en bloques.
    2. Guardrails se aplican a cada bloque:
       - max_oraciones: subdividir si excede.
       - max_palabras: subdividir si excede.
       - min_oraciones: fusionar con el bloque anterior si es muy chico.
"""

from __future__ import annotations

import logging

from .models import Oracion, Segmento

logger = logging.getLogger(__name__)


def agrupar_segmentos(
    oraciones: list[Oracion],
    breakpoints: list,
    min_oraciones: int = 2,
    max_oraciones: int = 8,
    max_palabras: int = 150,
    video_id: str = "",
) -> list[Segmento]:
    """Agrupa oraciones en segmentos semánticos coherentes.

    Args:
        oraciones: Lista de Oracion en orden cronológico.
        breakpoints: Lista de Breakpoint con .indice (cortar antes de esa oración).
        min_oraciones: Mínimo de oraciones por segmento (default: 2).
        max_oraciones: Máximo de oraciones por segmento (default: 8).
        max_palabras: Máximo de palabras por segmento (default: 150).
        video_id: ID del video de origen.

    Returns:
        Lista de Segmento en orden cronológico.
    """
    if not oraciones:
        return []

    # 1. Cortar por breakpoints → bloques de oraciones
    bloques = _cortar_por_breakpoints(oraciones, breakpoints)

    # 2. Subdividir bloques que exceden guardrails superiores
    bloques = _subdividir(bloques, max_oraciones, max_palabras)

    # 3. Fusionar bloques que no cumplen min_oraciones
    bloques = _fusionar_pequenos(bloques, min_oraciones)

    # 4. Construir Segmento de cada bloque
    segmentos: list[Segmento] = [
        _construir_segmento(bloque, video_id) for bloque in bloques
    ]

    logger.info(
        f"Agrupación: {len(oraciones)} oraciones → {len(segmentos)} segmentos"
    )
    return segmentos


# ──────────────────────────────────────────────────────────
# Paso 1: cortar por breakpoints
# ──────────────────────────────────────────────────────────

def _cortar_por_breakpoints(
    oraciones: list[Oracion],
    breakpoints: list,
) -> list[list[Oracion]]:
    """Divide la secuencia en bloques usando los breakpoints."""
    if not breakpoints:
        return [list(oraciones)]

    indices_corte = sorted(b.indice for b in breakpoints)

    bloques: list[list[Oracion]] = []
    inicio = 0

    for idx in indices_corte:
        if idx > inicio:
            bloques.append(oraciones[inicio:idx])
        inicio = idx

    if inicio < len(oraciones):
        bloques.append(oraciones[inicio:])

    return bloques if bloques else [list(oraciones)]


# ──────────────────────────────────────────────────────────
# Paso 2: subdividir bloques grandes
# ──────────────────────────────────────────────────────────

def _subdividir(
    bloques: list[list[Oracion]],
    max_oraciones: int,
    max_palabras: int,
) -> list[list[Oracion]]:
    """Subdivide los bloques que exceden max_oraciones o max_palabras."""
    resultado: list[list[Oracion]] = []

    for bloque in bloques:
        if _cumple_limite(bloque, max_oraciones, max_palabras):
            resultado.append(bloque)
        else:
            resultado.extend(
                _dividir_bloque(bloque, max_oraciones, max_palabras)
            )

    return resultado


def _cumple_limite(
    bloque: list[Oracion],
    max_oraciones: int,
    max_palabras: int,
) -> bool:
    """Verifica si un bloque respeta los límites superiores."""
    if len(bloque) > max_oraciones:
        return False
    return sum(len(o.words) for o in bloque) <= max_palabras


def _dividir_bloque(
    bloque: list[Oracion],
    max_oraciones: int,
    max_palabras: int,
) -> list[list[Oracion]]:
    """Divide un bloque grande en sub-bloques que respeten los límites."""
    sub_bloques: list[list[Oracion]] = []
    actual: list[Oracion] = []
    words_actual = 0

    for orac in bloque:
        words_orac = len(orac.words)
        excede_oraciones = len(actual) + 1 > max_oraciones
        excede_palabras = words_actual + words_orac > max_palabras

        if (excede_oraciones or excede_palabras) and actual:
            sub_bloques.append(actual)
            actual = []
            words_actual = 0

        actual.append(orac)
        words_actual += words_orac

    if actual:
        sub_bloques.append(actual)

    return sub_bloques


# ──────────────────────────────────────────────────────────
# Paso 3: fusionar bloques muy pequeños
# ──────────────────────────────────────────────────────────

def _fusionar_pequenos(
    bloques: list[list[Oracion]],
    min_oraciones: int,
) -> list[list[Oracion]]:
    """Fusiona bloques con menos de min_oraciones con el bloque anterior."""
    if not bloques:
        return []

    resultado: list[list[Oracion]] = [bloques[0]]

    for bloque in bloques[1:]:
        if len(bloque) < min_oraciones and resultado:
            resultado[-1].extend(bloque)
        else:
            resultado.append(bloque)

    return resultado


# ──────────────────────────────────────────────────────────
# Paso 4: construir Segmento
# ──────────────────────────────────────────────────────────

def _construir_segmento(
    bloque: list[Oracion],
    video_id: str,
) -> Segmento:
    """Construye un Segmento desde un bloque de oraciones."""
    return Segmento(
        texto=" ".join(o.texto for o in bloque),
        t_start=bloque[0].t_start,
        t_end=bloque[-1].t_end,
        oraciones=list(bloque),
        word_count=sum(len(o.words) for o in bloque),
        video_id=video_id,
    )
