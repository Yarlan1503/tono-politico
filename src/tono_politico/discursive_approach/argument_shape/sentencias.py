"""Extracción de oraciones desde ActorTranscript (sin word-level)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...speech2text.models import ActorTranscript, ActorTranscriptSegment
from .models import Oracion

if TYPE_CHECKING:
    from typing import Protocol

    class SpacyLike(Protocol):
        def __call__(self, text: str) -> Any: ...

        def pipe(self, texts: Any, batch_size: int = ...) -> Any: ...


logger = logging.getLogger(__name__)


def extraer_oraciones_de_transcript(
    transcript: ActorTranscript,
    nlp: SpacyLike,
) -> list[Oracion]:
    """Divide turnos del actor en oraciones con tiempos proporcionales."""
    if not transcript.segments:
        return []

    validos = [s for s in transcript.segments if s.text and s.text.strip()]
    if not validos:
        return []

    textos = [s.text for s in validos]
    docs = list(nlp.pipe(textos, batch_size=50))

    oraciones: list[Oracion] = []
    for seg, doc in zip(validos, docs, strict=True):
        oraciones.extend(_oraciones_de_turno(seg, doc))
    return oraciones


def _oraciones_de_turno(segment: ActorTranscriptSegment, doc: Any) -> list[Oracion]:
    sents = list(doc.sents)
    spans: list[tuple[str, int, int]] = []
    for sent in sents:
        text = sent.text.strip()
        if not text:
            continue
        spans.append((text, int(sent.start_char), int(sent.end_char)))

    if not spans:
        # fallback: un turno = una oración
        return [
            Oracion(
                texto=segment.text.strip(),
                t_start=segment.t_start,
                t_end=segment.t_end,
            )
        ]

    total_chars = sum(max(1, end - start) for _, start, end in spans)
    dur = max(0.0, segment.t_end - segment.t_start)
    t_cursor = segment.t_start
    result: list[Oracion] = []
    for i, (text, start, end) in enumerate(spans):
        frac = max(1, end - start) / total_chars
        t_len = dur * frac
        t_start = t_cursor
        t_end = segment.t_end if i == len(spans) - 1 else t_cursor + t_len
        result.append(Oracion(texto=text, t_start=t_start, t_end=t_end))
        t_cursor = t_end
    return result
