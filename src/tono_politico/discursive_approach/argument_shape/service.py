"""ArgumentShapeService: ActorTranscript → list[Argumento]."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...diarizacion.models import ActorTranscript
from .agrupacion import agrupar_argumentos
from .breakpoints import detectar_breakpoints
from .models import Argumento
from .sentencias import extraer_oraciones_de_transcript

if TYPE_CHECKING:
    from typing import Protocol

    class SpacyLike(Protocol):
        def __call__(self, text: str) -> Any: ...

        def pipe(self, texts: Any, batch_size: int = ...) -> Any: ...

    class EmbeddingLike(Protocol):
        def encode(self, texts: list[str]) -> list[list[float]]: ...


logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "LiquidAI/LFM2.5-Embedding-350M"


class ArgumentShapeService:
    """Forma argumentos semánticos a partir de transcripciones actor-only."""

    def __init__(
        self,
        spacy_model: str = "es_core_news_lg",
        embedding_model_name: str = EMBEDDING_MODEL,
        breakpoint_percentile: int = 95,
        min_oraciones: int = 2,
        max_oraciones: int = 8,
        max_palabras: int = 150,
    ) -> None:
        self.spacy_model_name = spacy_model
        self.embedding_model_name = embedding_model_name
        self.breakpoint_percentile = breakpoint_percentile
        self.min_oraciones = min_oraciones
        self.max_oraciones = max_oraciones
        self.max_palabras = max_palabras
        self._nlp: Any = None
        self._embedder: Any = None

    def procesar_one(self, transcript: ActorTranscript) -> list[Argumento]:
        """Procesa un único audio (no cruza videos)."""
        if not transcript.segments:
            return []

        nlp = self._get_nlp()
        embedder = self._get_embedder()
        oraciones = extraer_oraciones_de_transcript(transcript, nlp)
        if not oraciones:
            return []

        breakpoints = detectar_breakpoints(
            oraciones,
            embedder,
            self.breakpoint_percentile,
        )
        return agrupar_argumentos(
            oraciones,
            breakpoints,
            min_oraciones=self.min_oraciones,
            max_oraciones=self.max_oraciones,
            max_palabras=self.max_palabras,
            video_id=transcript.video_id,
            fecha=getattr(transcript, "fecha", None),
        )

    def procesar_corpus(self, transcripts: list[ActorTranscript]) -> list[Argumento]:
        """Procesa varios audios; cada uno por separado."""
        todos: list[Argumento] = []
        for tr in transcripts:
            todos.extend(self.procesar_one(tr))
        logger.info(
            "argument_shape: %s argumentos de %s transcripts",
            len(todos),
            len(transcripts),
        )
        return todos

    def _get_nlp(self) -> Any:
        if self._nlp is None:
            import spacy  # type: ignore[import-not-found]

            logger.info("Cargando spaCy: %s", self.spacy_model_name)
            self._nlp = spacy.load(self.spacy_model_name)
        return self._nlp

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            from sentence_transformers import (  # type: ignore[import-not-found]
                SentenceTransformer,
            )

            logger.info("Cargando embedder: %s", self.embedding_model_name)
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder
