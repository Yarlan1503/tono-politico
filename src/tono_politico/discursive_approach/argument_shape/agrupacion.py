"""Agrupa oraciones en Argumento con guardrails de tamaño."""

from __future__ import annotations

import logging
import re

from .breakpoints import Breakpoint
from .models import Argumento, Oracion

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\S+")


def contar_palabras(texto: str) -> int:
    """Cuenta tokens de texto (no depende de word-level ASR)."""
    return len(_WORD_RE.findall(texto))


def agrupar_argumentos(
    oraciones: list[Oracion],
    breakpoints: list[Breakpoint],
    min_oraciones: int = 2,
    max_oraciones: int = 8,
    max_palabras: int = 150,
    video_id: str = "",
    fecha: str | None = None,
) -> list[Argumento]:
    """Agrupa oraciones en argumentos semánticos coherentes."""
    if not oraciones:
        return []

    bloques = _cortar_por_breakpoints(oraciones, breakpoints)
    bloques = _subdividir(bloques, max_oraciones, max_palabras)
    bloques = _fusionar_pequenos(bloques, min_oraciones)
    argumentos = [_construir_argumento(b, video_id, fecha) for b in bloques]
    logger.info("Agrupación: %s oraciones → %s argumentos", len(oraciones), len(argumentos))
    return argumentos


def _cortar_por_breakpoints(
    oraciones: list[Oracion],
    breakpoints: list[Breakpoint],
) -> list[list[Oracion]]:
    if not breakpoints:
        return [list(oraciones)]
    indices = sorted(b.indice for b in breakpoints)
    bloques: list[list[Oracion]] = []
    inicio = 0
    for idx in indices:
        if idx > inicio:
            bloques.append(oraciones[inicio:idx])
        inicio = idx
    if inicio < len(oraciones):
        bloques.append(oraciones[inicio:])
    return bloques if bloques else [list(oraciones)]


def _palabras_bloque(bloque: list[Oracion]) -> int:
    return sum(contar_palabras(o.texto) for o in bloque)


def _cumple_limite(bloque: list[Oracion], max_oraciones: int, max_palabras: int) -> bool:
    if len(bloque) > max_oraciones:
        return False
    return _palabras_bloque(bloque) <= max_palabras


def _subdividir(
    bloques: list[list[Oracion]],
    max_oraciones: int,
    max_palabras: int,
) -> list[list[Oracion]]:
    resultado: list[list[Oracion]] = []
    for bloque in bloques:
        if _cumple_limite(bloque, max_oraciones, max_palabras):
            resultado.append(bloque)
        else:
            resultado.extend(_dividir_bloque(bloque, max_oraciones, max_palabras))
    return resultado


def _dividir_bloque(
    bloque: list[Oracion],
    max_oraciones: int,
    max_palabras: int,
) -> list[list[Oracion]]:
    sub: list[list[Oracion]] = []
    actual: list[Oracion] = []
    words_actual = 0
    for orac in bloque:
        w = contar_palabras(orac.texto)
        excede_o = len(actual) + 1 > max_oraciones
        excede_p = words_actual + w > max_palabras
        if (excede_o or excede_p) and actual:
            sub.append(actual)
            actual = []
            words_actual = 0
        actual.append(orac)
        words_actual += w
    if actual:
        sub.append(actual)
    return sub


def _fusionar_pequenos(
    bloques: list[list[Oracion]],
    min_oraciones: int,
) -> list[list[Oracion]]:
    if not bloques:
        return []
    resultado: list[list[Oracion]] = [bloques[0]]
    for bloque in bloques[1:]:
        if len(bloque) < min_oraciones and resultado:
            resultado[-1].extend(bloque)
        else:
            resultado.append(bloque)
    return resultado


def _construir_argumento(
    bloque: list[Oracion],
    video_id: str,
    fecha: str | None,
) -> Argumento:
    texto = " ".join(o.texto for o in bloque)
    return Argumento(
        texto=texto,
        t_start=bloque[0].t_start,
        t_end=bloque[-1].t_end,
        oraciones=list(bloque),
        word_count=contar_palabras(texto),
        video_id=video_id,
        fecha=fecha,
    )
