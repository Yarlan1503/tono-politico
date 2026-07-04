"""Componente 5: Tono — análisis de tono político."""

from .embeddings import EmbeddorTono, cosine_similarity, cosine_similarity_batch, mean_pooling
from .models import (
    EtiquetaScore,
    ResultadoEstiloDiscursivo,
    ResultadoFuncionDiscursiva,
    ResultadoLogicaPolitica,
    ResultadoSentimiento,
    ResultadoStance,
    ResultadoTono,
    SegmentoConTono,
)
from .service import TonoService, mapear_scores
from .taxonomia import prototipos_de, todas_las_dimensiones
from .zero_shot import ClasificadorLLM, construir_prompt_stance, parsear_stance

__all__ = [
    "ClasificadorLLM",
    "EmbeddorTono",
    "EtiquetaScore",
    "ResultadoEstiloDiscursivo",
    "ResultadoFuncionDiscursiva",
    "ResultadoLogicaPolitica",
    "ResultadoSentimiento",
    "ResultadoStance",
    "ResultadoTono",
    "SegmentoConTono",
    "TonoService",
    "clasificar_stance",
    "construir_prompt_stance",
    "cosine_similarity",
    "cosine_similarity_batch",
    "mapear_scores",
    "mean_pooling",
    "parsear_stance",
    "prototipos_de",
    "todas_las_dimensiones",
]
