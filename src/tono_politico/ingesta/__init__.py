"""Componente 1: Ingesta.

API pública:
    IngestaService — service OOP con config encapsulada.
    procesar_playlist — shortcut funcional (usa IngestaService con defaults).

Los módulos internos (playlist, audio, transcripcion, cache, service)
son la implementación.
"""

from ..models import VideoTranscript
from .service import IngestaService

__all__ = ["IngestaService", "VideoTranscript"]
