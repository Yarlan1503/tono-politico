"""Componente 2: Segmentación — service OOP.

Orquesta los 3 pasos:
    1. extraer_oraciones → spaCy divide SegmentoRaw en Oracion
    2. detectar_breakpoints → embeddings detectan cambios de tópico
    3. agrupar_segmentos → guardrails (min/max oraciones, max palabras)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..models import VideoTranscript
from .agrupacion import agrupar_segmentos
from .breakpoints import detectar_breakpoints
from .models import Segmento
from .sentencias import extraer_oraciones

if TYPE_CHECKING:
    from typing import Protocol

    class SpacyLike(Protocol):
        def __call__(self, text: str) -> Any: ...

    class EmbeddingLike(Protocol):
        def encode(self, texts: list[str]) -> list[list[float]]: ...

logger = logging.getLogger(__name__)


class SegmentacionService:
    """Service del Componente 2: segmentación semántica.

    Configura spaCy, el modelo de embeddings y los guardrails una sola vez.
    Recibe list[VideoTranscript] y devuelve list[Segmento].

    Attributes:
        spacy_model: Nombre del modelo spaCy (default: es_core_news_lg).
        breakpoint_percentile: Percentil de distancia coseno (default: 95).
        min_oraciones: Mínimo de oraciones por segmento (default: 2).
        max_oraciones: Máximo de oraciones por segmento (default: 8).
        max_palabras: Máximo de palabras por segmento (default: 150).
    """

    def __init__(
        self,
        spacy_model: str = "es_core_news_lg",
        breakpoint_percentile: int = 95,
        min_oraciones: int = 2,
        max_oraciones: int = 8,
        max_palabras: int = 150,
    ) -> None:
        self.spacy_model_name = spacy_model
        self.breakpoint_percentile = breakpoint_percentile
        self.min_oraciones = min_oraciones
        self.max_oraciones = max_oraciones
        self.max_palabras = max_palabras

        # Modelos lazy-load (se cargan en el primer procesar())
        self._nlp: SpacyLike | None = None
        self._embedder: EmbeddingLike | None = None

    def procesar(
        self, transcripts: list[VideoTranscript]
    ) -> list[Segmento]:
        """Segmenta transcripciones en bloques semánticamente coherentes.

        Args:
            transcripts: Lista de VideoTranscript del Componente 1.

        Returns:
            Lista de Segmento en orden cronológico.
        """
        if not transcripts:
            return []

        nlp = self._get_nlp()
        embedder = self._get_embedder()

        todos_segmentos: list[Segmento] = []

        for transcript in transcripts:
            if not transcript.raw_segments:
                logger.info(
                    f"Video {transcript.video_id} sin raw_segments, omitiendo"
                )
                continue

            # 1. Extraer oraciones
            oraciones = extraer_oraciones(transcript.raw_segments, nlp)
            if not oraciones:
                continue

            # 2. Detectar breakpoints semánticos
            breakpoints = detectar_breakpoints(
                oraciones, embedder, self.breakpoint_percentile
            )

            # 3. Agrupar en segmentos
            segmentos = agrupar_segmentos(
                oraciones,
                breakpoints,
                min_oraciones=self.min_oraciones,
                max_oraciones=self.max_oraciones,
                max_palabras=self.max_palabras,
                video_id=transcript.video_id,
            )

            todos_segmentos.extend(segmentos)

        logger.info(
            f"Segmentación completa: {len(todos_segmentos)} segmentos "
            f"de {len(transcripts)} videos"
        )
        return todos_segmentos

    def _get_nlp(self) -> Any:
        """Carga perezosa del modelo spaCy."""
        if self._nlp is None:
            import spacy  # type: ignore[import-not-found]

            logger.info(f"Cargando modelo spaCy: {self.spacy_model_name}")
            self._nlp = spacy.load(self.spacy_model_name)
        return self._nlp

    def _get_embedder(self) -> Any:
        """Carga perezosa del modelo de embeddings."""
        if self._embedder is None:
            from sentence_transformers import (  # type: ignore[import-not-found]
                SentenceTransformer,
            )

            logger.info("Cargando modelo: LiquidAI/LFM2.5-Embedding-350M")
            self._embedder = SentenceTransformer(
                "LiquidAI/LFM2.5-Embedding-350M"
            )
        return self._embedder
