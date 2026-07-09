"""TopicsClusterService: list[Argumento] → ResultadoTemas."""

from __future__ import annotations

import logging
from typing import Any

from ..argument_shape.models import Argumento
from .descubrimiento import descubrir_temas
from .models import ResultadoTemas

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "LiquidAI/LFM2.5-Embedding-350M"


class TopicsClusterService:
    def __init__(
        self,
        min_topic_size: int = 3,
        n_neighbors: int = 10,
        n_components: int = 5,
        embedding_model_name: str = EMBEDDING_MODEL,
    ) -> None:
        self.min_topic_size = min_topic_size
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.embedding_model_name = embedding_model_name
        self._embedder: Any = None

    def procesar(self, argumentos: list[Argumento]) -> ResultadoTemas:
        if not argumentos:
            logger.info("Sin argumentos para tematizar")
            return ResultadoTemas()
        return descubrir_temas(
            argumentos,
            embedding_model=self._get_embedder(),
            min_topic_size=self.min_topic_size,
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
        )

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            from sentence_transformers import (  # type: ignore[import-not-found]
                SentenceTransformer,
            )

            logger.info("Cargando embedder cluster: %s", self.embedding_model_name)
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder
