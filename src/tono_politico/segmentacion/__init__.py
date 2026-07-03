"""Componente 2: Segmentación.

API pública:
    SegmentacionService — service OOP con config encapsulada.
    Segmento, Oracion — DTOs de salida.

Pipeline:
    VideoTranscript[] → spaCy → embeddings → breakpoints → guardrails → Segmento[]
"""

from .models import Oracion, Segmento
from .service import SegmentacionService

__all__ = ["SegmentacionService", "Segmento", "Oracion"]
