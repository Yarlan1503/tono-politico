"""Tests para tono/embeddings.py — funciones puras y wrapper del modelo.

Los tests de funciones puras (mean_pooling, cosine_similarity) no requieren
cargar el modelo y cubren la lógica crítica.
El test de EmbeddorTono es de integración (marca slow).
"""

from __future__ import annotations

import numpy as np
import pytest

from tono_politico.tono.embeddings import (
    EmbeddorTono,
    cosine_similarity,
    cosine_similarity_batch,
    mean_pooling,
)


# ============================================================
# MEAN POOLING
# ============================================================
class TestMeanPooling:
    def test_vector_unitario_sin_padding(self):
        """Sin padding (mask toda 1), el resultado es el promedio de los tokens."""
        import torch

        # 2 tokens, 3 dims, sin padding
        token_embs = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        mask = torch.tensor([[1, 1]])
        result = mean_pooling(token_embs, mask)

        # Promedio: [0.5, 0.5, 0.0]
        expected = torch.tensor([[0.5, 0.5, 0.0]])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_ignora_tokens_con_padding(self):
        """Los tokens con mask=0 no deben contribuir al promedio."""
        import torch

        # 3 tokens pero el último es padding (mask=0)
        token_embs = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]])
        mask = torch.tensor([[1, 1, 0]])
        result = mean_pooling(token_embs, mask)

        # Solo cuenta los primeros 2: [0.5, 0.5]
        expected = torch.tensor([[0.5, 0.5]])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_todos_padding_devuelve_ceros(self):
        """Si todos los tokens son padding, devuelve ceros (evita división por cero)."""
        import torch

        token_embs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.tensor([[0, 0]])
        result = mean_pooling(token_embs, mask)

        assert torch.allclose(result, torch.zeros(1, 2), atol=1e-6)


# ============================================================
# COSINE SIMILARITY
# ============================================================
class TestCosineSimilarity:
    def test_vectores_iguales_devuelven_1(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_vectores_ortogonales_devuelven_0(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_vectores_opuestos_devuelven_menos_1(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_vector_cero_devuelve_0(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_no_normalizado_funciona(self):
        """La función debe normalizar internamente."""
        a = np.array([2.0, 0.0])  # magnitud 2
        b = np.array([3.0, 0.0])  # magnitud 3
        # Cosine sim = (6) / (2*3) = 1.0
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)


class TestCosineSimilarityBatch:
    def test_un_query_con_multiples_prototipos(self):
        query = np.array([1.0, 0.0])
        protos = np.array(
            [
                [1.0, 0.0],  # sim = 1.0
                [0.0, 1.0],  # sim = 0.0
                [-1.0, 0.0],  # sim = -1.0
            ]
        )
        result = cosine_similarity_batch(query, protos)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(1.0, abs=1e-6)
        assert result[1] == pytest.approx(0.0, abs=1e-6)
        assert result[2] == pytest.approx(-1.0, abs=1e-6)

    def test_query_2d_con_prototipos_2d(self):
        """Si el query es 2D (1, dim), debe funcionar igual."""
        query = np.array([[1.0, 0.0]])
        protos = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = cosine_similarity_batch(query, protos)
        assert result.shape == (2,)


# ============================================================
# EMBED_BATCH — batching real (no requiere modelo, usa fakes)
# ============================================================
class TestEmbedBatchReal:
    """Verifica que embed_batch hace una sola pasada del modelo para N textos."""

    def test_embed_batch_llama_modelo_una_sola_vez(self):
        import torch

        forward_calls = 0

        class FakeTokenizer:
            def __call__(self, texts, **kwargs):
                if isinstance(texts, str):
                    texts = [texts]
                n = len(texts)
                return {
                    "input_ids": torch.zeros((n, 4), dtype=torch.long),
                    "attention_mask": torch.ones((n, 4), dtype=torch.long),
                }

        class FakeModel:
            def __call__(self, **kwargs):
                nonlocal forward_calls
                forward_calls += 1
                batch_size = kwargs["input_ids"].shape[0]
                return type(
                    "Out",
                    (),
                    {"last_hidden_state": torch.randn(batch_size, 4, 8)},
                )()

            def eval(self):
                pass

        emb = EmbeddorTono()
        emb._tokenizer = FakeTokenizer()
        emb._model = FakeModel()

        result = emb.embed_batch(["hola", "mundo", "foo"])

        assert forward_calls == 1, f"Esperaba 1 forward, got {forward_calls}"
        assert result.shape == (3, 8)

    def test_embed_batch_lista_vacia_devuelve_shape_0(self):
        emb = EmbeddorTono()

        result = emb.embed_batch([])
        assert result.shape == (0,)


# ============================================================
# EMBEDDOR TONO (integración — requiere modelo)
# ============================================================
@pytest.mark.slow
class TestEmbeddorTono:
    """Tests de integración que cargan el modelo real.

    Se ejecutan solo con: pytest tests/test_tono_embeddings.py -m slow
    """

    def test_embed_produce_vector_no_degenerado(self):
        emb = EmbeddorTono()
        v1 = emb.embed("El pueblo exige justicia social.")
        v2 = emb.embed("La ciencia demuestra que el fracking contamina.")

        assert v1.shape == v2.shape
        assert v1.shape[0] > 100  # LFM2.5 produce 1024 dims

        sim = cosine_similarity(v1, v2)
        # No deben ser idénticos ( SentenceTransformer sí los hacía idénticos)
        assert sim < 0.99, (
            f"Embeddings degenerados: sim={sim:.4f} — el mean pooling no está funcionando"
        )

    def test_embed_batch_consistente_con_individual(self):
        emb = EmbeddorTono()
        textos = ["Hola mundo", "Adiós mundo"]
        batch = emb.embed_batch(textos)
        individual = np.array([emb.embed(t) for t in textos])

        assert batch.shape == individual.shape
        assert np.allclose(batch, individual, atol=1e-5)
