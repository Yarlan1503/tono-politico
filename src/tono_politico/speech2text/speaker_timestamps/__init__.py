"""speaker_timestamps: quién habla cuándo + match del actor (sin ASR)."""

from __future__ import annotations

from .models import PerfilVozActor, SpeakerMatch, TurnoOrador
from .service import SpeakerTimestampsService, fusionar_turnos_consecutivos

__all__ = [
    "SpeakerTimestampsService",
    "fusionar_turnos_consecutivos",
    "TurnoOrador",
    "PerfilVozActor",
    "SpeakerMatch",
]
