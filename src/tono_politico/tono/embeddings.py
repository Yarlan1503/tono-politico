"""Wrapper de LFM2.5-Embedding-350M con mean pooling correcto.

IMPORTANTE: No usamos sentence-transformers para este modelo porque su
wrapper produce embeddings degenerados (todas las similitudes = 1.0) con
LFM2.5-Embedding-350M. En su lugar, cargamos el modelo directamente con
transformers (AutoModel) y aplicamos mean pooling manualmente sobre
last_hidden_state con attention_mask.

Funciones puras:
- mean_pooling(token_embs, mask) → vector promedio
- cosine_similarity(a, b) → float
- cosine_similarity_batch(query, prototipos) → np.array

Clase:
- EmbeddorTono: carga perezosa del modelo y expone embed() / embed_batch()
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Modelo de embeddings compartido con Componentes 2 y 3
EMBEDDING_MODEL = "LiquidAI/LFM2.5-Embedding-350M"


def mean_pooling(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Promedia los embeddings de tokens con mean pooling.

    Usa attention_mask para ignorar tokens de padding.

    Args:
        token_embeddings: Tensor de shape (batch, seq_len, dim).
        attention_mask: Tensor de shape (batch, seq_len) con 1s y 0s.

    Returns:
        Tensor de shape (batch, dim) con los embeddings promedio.
        No normaliza L2 — la normalización va en cosine_similarity.
    """
    # Expandir mask a (batch, seq_len, 1) para multiplicar
    mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)

    # Sumar embeddings donde mask=1
    summed = (token_embeddings * mask).sum(dim=1)  # (batch, dim)

    # Contar tokens válidos (clamped para evitar división por cero)
    counts = mask.sum(dim=1).clamp(min=1e-9)  # (batch, 1)

    # Promedio
    pooled = summed / counts  # (batch, dim)

    return pooled


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud coseno entre dos vectores 1D.

    Normaliza internamente; no requiere que los vectores estén pre-normalizados.

    Args:
        a: Vector 1D.
        b: Vector 1D.

    Returns:
        Similitud coseno en [-1.0, 1.0]. 0.0 si algún vector es cero.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_batch(
    query: np.ndarray,
    prototipos: np.ndarray,
) -> np.ndarray:
    """Similitud coseno entre un vector query y un batch de prototipos.

    Args:
        query: Vector 1D de shape (dim,) o 2D de shape (1, dim).
        prototipos: Matriz de shape (n_prototipos, dim).

    Returns:
        Array 1D de shape (n_prototipos,) con las similitudes.
    """
    if query.ndim == 2:
        query = query.squeeze(0)

    # Normalizar
    q_norm = np.linalg.norm(query)
    p_norms = np.linalg.norm(prototipos, axis=1)

    # Evitar división por cero
    if q_norm == 0:
        return np.zeros(len(prototipos))
    p_norms_safe = np.where(p_norms == 0, 1e-9, p_norms)

    dots = prototipos @ query
    return dots / (p_norms_safe * q_norm)


class EmbeddorTono:
    """Wrapper de LFM2.5-Embedding-350M con mean pooling correcto.

    Carga perezosa: el modelo se carga en el primer embed()/embed_batch().

    Attributes:
        model_name: Nombre del modelo en HuggingFace.
        device: "cpu" o "cuda".
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Carga perezosa del modelo y tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Cargando modelo: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
        )
        self._model.eval()
        self._model.to(self.device)

    def embed(self, text: str) -> np.ndarray:
        """Embebe un texto y devuelve un vector 1D.

        Wrapper sobre embed_batch para un solo texto.
        """
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embebe múltiples textos en una sola pasada del modelo.

        Tokeniza todos los textos juntos con padding dinámico y procesa
        el batch completo en un solo forward pass.

        Args:
            texts: Lista de textos a embeber.

        Returns:
            Matriz de shape (n_texts, dim). Lista vacía → shape (0,).
        """
        if not texts:
            return np.zeros((0,))

        self._load()
        assert self._tokenizer is not None
        assert self._model is not None

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        embs = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        return embs.float().cpu().numpy()
