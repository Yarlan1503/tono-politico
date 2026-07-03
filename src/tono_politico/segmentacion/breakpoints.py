"""Detección de breakpoints semánticos entre oraciones.

Sigue el estándar LangChain SemanticChunker / LlamaIndex:
codifica todas las oraciones, calcula distancia coseno entre consecutivas,
y marca breakpoint donde la distancia supera el percentil indicado.

La segmentación acústica la hace Whisper internamente al dividir el audio
en ventanas con timestamps. Este módulo se enfoca exclusivamente en la
señal semántica: detectar cambios de tópico entre oraciones.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from .models import Oracion

if TYPE_CHECKING:
    from typing import Protocol

    class EmbeddingLike(Protocol):
        def encode(self, texts: list[str]) -> list[list[float]]: ...

logger = logging.getLogger(__name__)


class Breakpoint:
    """Punto de corte semántico en la secuencia de oraciones.

    Attributes:
        indice: Cortar ANTES de esta oración (0-indexed).
        intensidad: Distancia coseno entre las dos oraciones que separa.
    """

    def __init__(self, indice: int, intensidad: float):
        self.indice = indice
        self.intensidad = intensidad

    def __repr__(self) -> str:
        return (
            f"Breakpoint(indice={self.indice}, "
            f"intensidad={self.intensidad:.3f})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Breakpoint):
            return NotImplemented
        return (
            self.indice == other.indice
            and abs(self.intensidad - other.intensidad) < 1e-6
        )


def detectar_breakpoints(
    oraciones: list[Oracion],
    model: EmbeddingLike,
    breakpoint_percentile: int = 95,
) -> list[Breakpoint]:
    """Detecta breakpoints semánticos en una secuencia de oraciones.

    Sigue el estándar LangChain/LlamaIndex: usa distancia coseno entre
    embeddings de oraciones consecutivas y corta donde la distancia supera
    el percentil indicado (default 95).

    Args:
        oraciones: Lista de Oracion en orden cronológico.
        model: Modelo de embeddings con .encode(list[str]) → list[vector].
        breakpoint_percentile: Percentil de distancia por encima del cual
            se genera un breakpoint (default: 95, como LangChain).

    Returns:
        Lista de Breakpoint ordenada por índice.
    """
    if len(oraciones) < 3:
        # Con < 3 oraciones no hay suficiente información estadística
        # para que el percentil sea significativo.
        return []

    textos = [o.texto for o in oraciones]
    embeddings = model.encode(textos)

    # Distancias coseno entre oraciones consecutivas
    distancias: list[float] = []
    for i in range(len(embeddings) - 1):
        dist = _cosine_distance(embeddings[i], embeddings[i + 1])
        distancias.append(dist)

    if len(distancias) < 2:
        return []

    # Umbral: percentil de las distancias observadas
    umbral = _percentil(distancias, breakpoint_percentile)

    # Margen mínimo para evitar falsos positivos por ruido numérico
    EPSILON = 1e-4

    breakpoints: list[Breakpoint] = []
    for i, dist in enumerate(distancias):
        if dist >= umbral and dist > EPSILON:
            breakpoints.append(Breakpoint(indice=i + 1, intensidad=dist))

    return breakpoints


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Distancia coseno = 1 - similitud coseno. Rango [0, 2]."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 1.0

    return 1.0 - (dot / (norm_a * norm_b))


def _percentil(valores: list[float], p: int) -> float:
    """Percentil p de una lista de valores (interpolación lineal)."""
    if not valores:
        return 0.0

    ordenados = sorted(valores)
    n = len(ordenados)
    k = (p / 100) * (n - 1)
    f = int(k)
    c = k - f

    if f + 1 < n:
        return ordenados[f] + c * (ordenados[f + 1] - ordenados[f])
    return ordenados[f]
