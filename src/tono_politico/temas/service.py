"""Componente 3: Temas — service OOP.

Descubre tópicos predominantes usando BERTopic con embeddings LFM2.5.
Configura BERTopic (UMAP + HDBSCAN + c-TF-IDF) en el constructor y
ejecuta fit_transform en procesar().
"""

from __future__ import annotations

import logging
from typing import Any

from ..segmentacion.models import Segmento
from .descubrimiento import descubrir_temas
from .models import ResultadoTemas

logger = logging.getLogger(__name__)

# Modelo de embeddings compartido con Componente 2
EMBEDDING_MODEL = "LiquidAI/LFM2.5-Embedding-350M"


class TemasService:
    """Service del Componente 3: descubrimiento de temas con BERTopic.

    Attributes:
        min_topic_size: Mínimo de segmentos por tópico (default: 3).
        n_neighbors: UMAP n_neighbors (default: 10).
        n_components: UMAP dimensiones (default: 5).
        embedding_model_name: Modelo de embeddings (default: LFM2.5).
    """

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

        # Lazy load
        self._embedder: Any = None

    def procesar(self, segmentos: list[Segmento]) -> ResultadoTemas:
        """Descubre temas predominantes en los segmentos.

        Args:
            segmentos: Lista de Segmento del Componente 2.

        Returns:
            ResultadoTemas con segmentos tematizados y metadata de tópicos.
        """
        if not segmentos:
            logger.info("Sin segmentos para tematizar")
            return ResultadoTemas()

        embedder = self._get_embedder()

        return descubrir_temas(
            segmentos,
            embedding_model=embedder,
            min_topic_size=self.min_topic_size,
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
        )

    def _get_embedder(self) -> Any:
        """Carga perezosa del modelo de embeddings LFM2.5."""
        if self._embedder is None:
            from sentence_transformers import (  # type: ignore[import-not-found]
                SentenceTransformer,
            )

            logger.info(f"Cargando modelo: {self.embedding_model_name}")
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder
