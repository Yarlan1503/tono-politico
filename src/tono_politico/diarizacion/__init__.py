"""Componente 1.5: Diarización e identificación de actor.

API pública:
    DiarizacionService — service OOP con config encapsulada.
    TurnoOrador, PerfilVozActor, SpeakerMatch — DTOs del componente.

Los módulos internos (diarizacion, perfil_voz, matching, alineacion)
son la implementación.
"""

from .models import PerfilVozActor, SpeakerMatch, TurnoOrador
from .service import DiarizacionService

__all__ = [
    "DiarizacionService",
    "TurnoOrador",
    "PerfilVozActor",
    "SpeakerMatch",
]
