"""Diarización e identificación de actor — DTOs y utilidades.

API pública:
    ActorTranscript, ActorTranscriptSegment, AsrMetadata — DTOs del transcript.
    TurnoOrador, PerfilVozActor, SpeakerMatch — DTOs del componente.

Los módulos internos (adapter, perfil_voz, matching, transcripcion_actor,
whisper_clip, actor_transcript) son la implementación reusada por speech2text.
"""

from .models import (
    ActorTranscript,
    ActorTranscriptSegment,
    AsrMetadata,
    PerfilVozActor,
    SpeakerMatch,
    TurnoOrador,
)

__all__ = [
    "TurnoOrador",
    "AsrMetadata",
    "ActorTranscriptSegment",
    "ActorTranscript",
    "PerfilVozActor",
    "SpeakerMatch",
]
