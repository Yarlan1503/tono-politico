"""Breakpoints semánticos (LangChain/LlamaIndex: coseno + percentil)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

from .models import Oracion

if TYPE_CHECKING:

    class EmbeddingLike(Protocol):
        def encode(self, texts: list[str]) -> list[list[float]]: ...


class Breakpoint:
    """Corte semántico: cortar ANTES de ``indice``."""

    def __init__(self, indice: int, intensidad: float) -> None:
        self.indice = indice
        self.intensidad = intensidad

    def __repr__(self) -> str:
        return f"Breakpoint(indice={self.indice}, intensidad={self.intensidad:.3f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Breakpoint):
            return NotImplemented
        return self.indice == other.indice and abs(self.intensidad - other.intensidad) < 1e-6


def detectar_breakpoints(
    oraciones: list[Oracion],
    model: EmbeddingLike,
    breakpoint_percentile: int = 95,
) -> list[Breakpoint]:
    """Detecta cortes donde la distancia coseno supera el percentil."""
    if len(oraciones) < 3:
        return []

    textos = [o.texto for o in oraciones]
    embeddings = model.encode(textos)

    distancias: list[float] = []
    for i in range(len(embeddings) - 1):
        distancias.append(_cosine_distance(embeddings[i], embeddings[i + 1]))

    if len(distancias) < 2:
        return []

    umbral = _percentil(distancias, breakpoint_percentile)
    epsilon = 1e-4
    breakpoints: list[Breakpoint] = []
    for i, dist in enumerate(distancias):
        if dist >= umbral and dist > epsilon:
            breakpoints.append(Breakpoint(indice=i + 1, intensidad=dist))
    return breakpoints


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - (dot / (norm_a * norm_b))


def _percentil(valores: list[float], p: int) -> float:
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
