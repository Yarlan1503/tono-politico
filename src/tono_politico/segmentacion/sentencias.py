"""Extracción de oraciones desde segmentos crudos de Whisper.

Usa spaCy para dividir SegmentoRaw en Oracion, preservando los
WordTimestamp de cada oración mediante mapeo por offsets de caracteres.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..models import SegmentoRaw, WordTimestamp
from .models import Oracion

if TYPE_CHECKING:
    from typing import Protocol

    class SpacyLike(Protocol):
        def __call__(self, text: str) -> Any: ...

logger = logging.getLogger(__name__)


def extraer_oraciones(
    segmentos: list[SegmentoRaw],
    nlp: SpacyLike,
) -> list[Oracion]:
    """Divide segmentos crudos en oraciones preservando timestamps.

    Usa spaCy para detectar límites de oración dentro del texto de cada
    SegmentoRaw. Luego mapea los WordTimestamp de Whisper a cada oración
    usando los offsets de caracteres que spaCy provee (sent.start_char /
    sent.end_char) contra la posición acumulada de cada palabra.

    Args:
        segmentos: Lista de SegmentoRaw (salida de Whisper).
        nlp: Instancia de modelo spaCy (o compatible con __call__ → doc).

    Returns:
        Lista de Oracion en orden cronológico.
    """
    if not segmentos:
        return []

    oraciones: list[Oracion] = []

    for segmento in segmentos:
        if not segmento.words:
            logger.warning(
                f"Segmento sin words, omitiendo: {segmento.texto[:50]}..."
            )
            continue

        oraciones.extend(
            _dividir_segmento(segmento, nlp)
        )

    return oraciones


def _dividir_segmento(
    segmento: SegmentoRaw, nlp: SpacyLike
) -> list[Oracion]:
    """Divide un SegmentoRaw individual en oraciones con words asignadas."""
    doc = nlp(segmento.texto)
    sents = list(doc.sents)

    # Construir mapa de offsets: para cada palabra, su rango [char_start, char_end)
    word_offsets = _calcular_word_offsets(segmento.texto, segmento.words)

    resultado: list[Oracion] = []

    for sent in sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        # Encontrar qué words caen dentro del rango de caracteres de esta oración
        sent_start = sent.start_char
        sent_end = sent.end_char

        words_desta_oracion = [
            w
            for i, (w_start, w_end, w) in enumerate(word_offsets)
            if w_start < sent_end and w_end > sent_start
        ]

        if not words_desta_oracion:
            continue

        resultado.append(
            Oracion(
                texto=sent_text,
                t_start=words_desta_oracion[0].start,
                t_end=words_desta_oracion[-1].end,
                words=words_desta_oracion,
            )
        )

    return resultado


def _calcular_word_offsets(
    texto: str, words: list[WordTimestamp]
) -> list[tuple[int, int, WordTimestamp]]:
    """Mapea cada WordTimestamp a su posición [start, end) en el texto.

    Busca cada palabra secuencialmente en el texto acumulando el offset
    para que las apariciones repetidas no causen falsos matches.
    """
    offsets: list[tuple[int, int, WordTimestamp]] = []
    pos = 0

    for w in words:
        clean_word = w.word.strip()
        if not clean_word:
            continue

        # Buscar la palabra desde la posición actual
        found = _find_word(texto, clean_word, pos)
        if found is None:
            # Fallback: búsqueda desde el inicio
            found = _find_word(texto, clean_word, 0)

        if found is None:
            # No encontrada: usar la posición acumulada como aproximación
            offsets.append((pos, pos + len(clean_word), w))
            pos += len(clean_word) + 1
        else:
            offsets.append((found, found + len(clean_word), w))
            pos = found + len(clean_word)

    return offsets


def _find_word(text: str, word: str, start: int) -> int | None:
    """Busca word en text desde start, case-insensitive, sin espacios."""
    return _find_ci(text, word, start)


def _find_ci(text: str, word: str, start: int) -> int | None:
    """Búsqueda case-insensitive desde un offset."""
    text_lower = text.lower()
    word_lower = word.lower().strip()
    idx = text_lower.find(word_lower, start)
    return idx if idx != -1 else None
