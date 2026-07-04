"""Componente 3: Temas.

API pública:
    TemasService — service OOP con BERTopic + LFM2.5-Embedding-350M.
    ResultadoTemas, TopicoInfo, SegmentoTematizado — DTOs de salida.

Pipeline:
    Segmento[] → BERTopic → ResultadoTemas (tópicos + asignaciones)
"""

from .models import ResultadoTemas, SegmentoTematizado, TopicoInfo
from .service import TemasService

__all__ = [
    "TemasService",
    "ResultadoTemas",
    "TopicoInfo",
    "SegmentoTematizado",
]
